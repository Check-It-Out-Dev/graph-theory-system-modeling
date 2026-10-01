# Redis unavailable for an hour in production: what breaks today, and how to degrade gracefully at runtime

## Problem
If Redis goes down for an hour while production is running, the main damage is that signed-in users get locked out. Every private API call returns 401, and the frontend then signs everyone out. Rate limiting and the geo-IP cache degrade quietly. Step-up checks, email verification and parts of upload-quota tracking fail with errors, and the health endpoint goes DOWN.

The `no-redis` profile does not help here. It only chooses the in-memory beans once, at startup. Done means: Redis failing at runtime moves each feature into a defined degraded mode, security guarantees still hold, Redis coming back is handled, and tests cover it.

## Where it lives today

**How the storage mode is chosen at startup**
- `backend/src/main/java/com/sm/instagram/platform/config/StorageModeConfiguration.java`: `storage.mode` is `redis` (the default) or `in-memory`.
  - `application.yml` and `application-prod.yml` set `redis`; `application-no-redis.yml` sets `in-memory`.
  - Each implementation switches on this property with `@ConditionalOnProperty`:
    - `RedisRateLimiterService` / `InMemoryRateLimiterService`
    - `RedisUserCache` / `InMemoryUserCache`
    - `RedisGeoLocationCache` / `InMemoryGeoLocationCache`
    - `StorageRateLimitService` / `InMemoryStorageRateLimitService`
    - `RedisValidationService`
    - `NoOpRedisHealthIndicator`
- `config/RedisConfiguration.java` is active unless `spring.data.redis.enabled=false`. It builds the Lettuce factory (standalone, sentinel or cluster), the `RedisTemplate`s and a `RedisCacheManager`. There is no `CacheErrorHandler`.
  - Lettuce is set to `REJECT_COMMANDS` while disconnected, with auto-reconnect.
  - Production uses sentinel, a 1000 ms command timeout and pool `max-wait` 2000 ms.
- `application-no-redis.yml` does five things: sets `storage.mode: in-memory`, sets `spring.data.redis.enabled: false`, excludes the Redis auto-configurations, sets `cache.type: simple`, and sets `management.health.redis.enabled: false`.
- `config/UnifiedStorageConfiguration.java` only logs which mode was chosen.
- `storage/config/StorageConfiguration.java` adds an in-memory `StorageRateLimitService` if no other bean exists.

**Rate limiting**
- Flow: `RateLimitConfiguration` registers `RateLimitInterceptor` → `GdprCompliantRateLimiterService` (`@Primary`), which picks its delegate once, in its constructor → `RedisRateLimiterService`.
- `RedisRateLimiterService` runs a Lua sliding-window script inside a Resilience4j circuit breaker. Any exception falls back to its own in-memory windows (`fallback-to-memory: true` in prod).

**User cache (used by authentication and every enforcement filter)**
- `auth/cache/UserCacheService.java` is the interface, implemented by `RedisUserCache` and `InMemoryUserCache`, both backed by `UserRepository`.
- Readers:
  - `common/authorization/JwtAuthenticationFilter.java`: `isUserActive`, `getTokenVersion`
  - `BannedUserAuthorizationFilter` and `ConsentEnforcementFilter`: `getAccountStatus`
  - `EmailVerificationEnforcementFilter`: `getEmailVerified`
- About 15 services and controllers call `evict` after changing a user: `UserService`, `LegalConsentService`, `EmailChangeService`, `TwoFactorStatusController`, `AdminCascadeDeleteServiceImpl`, `RegistryLookupService`, `UserAccountOrchestrator`, `FirebaseAuthProxyService`, `InstagramDataDeletionService` and others.

**Code that calls Redis directly, with no storage-mode switch**
- `auth/stepup/StepUpAuthService.java`: `StringRedisTemplate` for codes, attempts, cooldown, lockout and step-up tokens.
- `auth/service/EmailVerificationService.java`: `StringRedisTemplate` for the verification-code mapping.
- `@Cacheable` on `city/CityRepository.findByName` and `auth/service/RecaptchaService`.

**Geo-IP and sessions**
- `SessionSecurityService` → `GeoLocationService` → `common/security/geoip/GeoLocationFacade` → `GeoLocationCache` (Redis or in-memory) plus MaxMind.
- `SessionSecurityService` itself does not use Redis.
- `auth/session/SocialAuthSessionService` is always an in-memory `ConcurrentHashMap`.

**Uploads**
- `storage/service/StorageRateLimitService.java` keeps per-user hourly and daily sorted sets and a storage-usage counter in Redis.
- `storage/health/UploadSystemHealthIndicator.java` probes Redis on each health check.

**Health and startup**
- The production container healthcheck (`ansible/roles/11-docker-compose/templates/docker-compose-prod.yml.j2`) calls the overall `/api/actuator/health`. `application-prod.yml` enables `management.health.redis`.
- `config/RedisStartupConnectivityTest.java` is enabled in prod.

### What happens during a one-hour outage (app already running)

| Area | Outcome | Why |
|---|---|---|
| Authenticated private API | **Fails: 401, then mass sign-out** | `RedisUserCache.isUserActive` calls `redisTemplate.opsForValue().get` with no try/catch. `JwtAuthenticationFilter` catches any exception and returns 401 `error.auth.invalid_token`. The frontend `error.interceptor.ts` then calls `/auth/refresh-session`, which is not public, gets 401 again, treats that as a hard failure, clears the session and redirects to sign-in. |
| Public or optional-auth endpoints for signed-in users | Degrades to anonymous | The same catch branch clears the security context and continues. |
| Step-up (email change, sensitive actions, admin TOTP token) | **Fails with 500** | Direct `StringRedisTemplate` calls, no handling. |
| Sending and applying email verification | **Fails** | `storeOobCode` and `lookupOobCode` are not wrapped. |
| Recaptcha-protected actions, city lookup by name | **Likely fails** | `@Cacheable` goes to Redis and no `CacheErrorHandler` is configured (INFERENCE). |
| Admin or user changes that evict the cache (role change, consent, 2FA disable, cascade delete) | **Likely fails and rolls back** | `RedisUserCache.evict` is not wrapped. Whether it rolls back depends on each caller's transaction (INFERENCE). |
| API rate limiting | Degrades | Falls back to in-memory windows on every error, even before the breaker opens. Counts are per instance and reset when the outage starts. |
| Content-upload rate limit | Degrades (allows the upload) | The Lua call is inside a try block that returns `allowed(999,999)`. |
| Profile-photo upload quota, `recordUpload`, file-delete quota | **Fails** | `hasStorageSpace` is not wrapped; `recordUpload` and `decreaseUserStorage` rethrow as `StorageTranslatableException`. |
| Geo-IP lookup and impossible-travel check | Degrades | `RedisGeoLocationCache.get`/`put` swallow errors, so every lookup goes to MaxMind. The travel check has a 500 ms timeout: it lets the request through for most roles and blocks high-privilege roles after repeated failures. The MaxMind database update fails, because `tryLock` is not wrapped, so the old database stays. |
| Session fingerprint check, social-auth sessions | Keep working | No Redis involved. |
| `/actuator/health` | **DOWN (503)** | `UploadSystemHealthIndicator` reports DOWN whenever Redis is configured but unreachable, and the Redis health indicator is enabled in prod. The container is marked unhealthy after 20 × 60 s. `restart: unless-stopped` does not restart an unhealthy container. |
| Restart during the outage | Unclear | `RedisStartupConnectivityTest` only fails fast if `System.getProperty("spring.profiles.active")` contains `prod`. Production sets the profile through the `SPRING_PROFILES_ACTIVE` environment variable, and the `redis.startup.test.fail-fast: true` key in prod config is never read. So startup probably logs a warning and continues (HYPOTHESIS). |

## Proposed change
Keep `storage.mode` as the startup choice of which backends exist. In `redis` mode, make Redis the primary store and the in-memory implementations a runtime fallback, both behind the interfaces that already exist. This follows the pattern `RedisRateLimiterService` already uses (Resilience4j breaker plus in-memory fallback), spread to every Redis consumer and driven by one shared availability signal.

1. **One availability signal: `common/redis/RedisAvailability`.**
   - A Resilience4j `CircuitBreaker` fed by command failures, plus a scheduled PING.
   - Exposes `isAvailable()`, state-change events and a Micrometer gauge.
   - Replaces the private breaker inside `RedisRateLimiterService`.
2. **Decorators that switch over at runtime.** Each consumer gets a failure policy chosen by what it protects:
   - **User cache → read through to the database (fail open to the source of truth).**
     - A `ResilientUserCache` (`@Primary`) wraps `RedisUserCache`. When Redis is unavailable or throws, it serves from an `InMemoryUserCache` with a short TTL (about 30 s) that reads `UserRepository`.
     - Bans and `tokenVersion` bumps stay correct, because the database is authoritative.
     - A failed Redis `evict` never throws. The uid goes into a pending set that is flushed, or `evictAll()` runs, when Redis comes back.
   - **API rate limiting → in-memory per instance.** Keep this, and optionally divide limits by an `instance-count` property while degraded.
   - **Email verification codes → failover store.** Codes are written to memory when Redis is down. Reads check memory first, then Redis.
   - **Step-up → fail closed with 503** and a translatable "temporarily unavailable" error. Its attempt, cooldown and lockout counters are brute-force protection, and local counters that reset silently would weaken them.
   - **Upload quota → allow the upload and record a metric.** Quota updates are best effort and must not throw.
   - **Geo cache lock → `false` on error**, so the database update is skipped and retried on the next run.
   - **Spring Cache → `CacheErrorHandler` that logs and treats errors as a miss.**
3. **A new port for the direct `StringRedisTemplate` users.**
   - `EphemeralStore` offers `set` with TTL, `get`, `getAndDelete`, `increment` with TTL, `exists` and `delete`. It has Redis, in-memory and failover implementations.
   - `StepUpAuthService` and `EmailVerificationService` use it. This also removes their hard dependency on Redis, which may be why `no-redis` does not start today (HYPOTHESIS below).
4. **Infrastructure errors must not look like auth errors.** `JwtAuthenticationFilter` returns 503 for data-access or Redis exceptions instead of 401. The frontend interceptor only signs users out after a 401 from the refresh call, so a 503 leaves sessions alone.
5. **Health and startup reflect "degraded", not "down".**
   - Redis leaves the overall and readiness status and appears only as a detail.
   - `UploadSystemHealthIndicator` reports Redis as a degraded detail.
   - The container healthcheck moves to `/actuator/health/liveness`.
   - `RedisStartupConnectivityTest` reads the `fail-fast` property, set to false in prod once the fallbacks exist, so the app can start while Redis is down.

**Alternative considered: move verification codes and step-up state to Postgres.** It is durable and shared across instances, but needs a Liquibase changeset, a cleanup job and adds load on the write path. In-memory failover is enough if production runs one app container, which the compose template suggests (HYPOTHESIS). Choose the in-memory failover now, and revisit if the app scales out.

## Plan
1. **Observability and health first (no behaviour change).**
   - Add `backend/src/main/java/com/sm/instagram/platform/common/redis/RedisAvailability.java` and its Micrometer gauge.
   - `storage/health/UploadSystemHealthIndicator.java`: a Redis failure becomes a detail instead of DOWN.
   - `health/NoOpRedisHealthIndicator.java`: drop `matchIfMissing = true`.
   - `application-prod.yml`, `application-monitoring.yml`, `application-actuator.yml`: take Redis out of readiness and overall status.
   - `ansible/roles/11-docker-compose/templates/docker-compose-prod.yml.j2`: healthcheck → `/api/actuator/health/liveness`.
2. **Cache error handler.** `config/RedisConfiguration.java`: implement `CachingConfigurer` with a logging `CacheErrorHandler`.
3. **Resilient user cache.**
   - New `auth/cache/ResilientUserCache.java` and `auth/cache/UserCacheConfiguration.java`. Move the `storage.mode` conditions into `@Bean` methods so `InMemoryUserCache` can also be built as the fallback in Redis mode, with a configurable TTL.
   - `RedisUserCache.java`: throw a typed exception that the decorator catches.
   - Clear pending evictions when Redis comes back.
   - The ~15 callers stay unchanged because they depend on the interface.
4. **Filter status codes.** `common/authorization/JwtAuthenticationFilter.java`: data-access or Redis exceptions → 503, on private endpoints only; public endpoints keep degrading to anonymous.
5. **Rate limiter.**
   - `common/ratelimit/RedisRateLimiterService.java`: use `RedisAvailability`, skip Redis while it is open instead of paying a timeout per call, and make the fail-closed branch explicit.
   - `application-prod.yml`: optional `rate-limit.redis.fallback.instance-count`.
6. **`EphemeralStore` port.**
   - New `common/redis/EphemeralStore.java` with `RedisEphemeralStore`, `InMemoryEphemeralStore` and `FailoverEphemeralStore`.
   - Refactor `auth/service/EmailVerificationService.java` (failover) and `auth/stepup/StepUpAuthService.java` (fail closed: new translatable 503 key in `messages_en.properties` / `messages_pl.properties`).
7. **Upload quota.** `storage/service/StorageRateLimitService.java`: wrap `hasStorageSpace`, `updateUserStorage`, `decreaseUserStorage` and `recordUpload`; allow the upload, log and count the failure, never throw on quota bookkeeping.
8. **Geo cache.** `common/security/geoip/RedisGeoLocationCache.java`: wrap `evict`, `tryLock` (return false) and `releaseLock`.
9. **Startup.** `config/RedisStartupConnectivityTest.java`: replace the `System.getProperty` profile check with `@Value("${redis.startup.test.fail-fast:false}")`. Set prod to `false` only after steps 3–8 are deployed.
10. **Frontend (optional, UX only).** `frontend/src/app/core/interceptors/error.interceptor.ts` already ignores 5xx for sign-out. Add a "temporarily degraded" toast or banner for the new 503 key, possibly through `core/shell/shell-status.service.ts` (HYPOTHESIS about that service's role).
11. **Tests** (see below). Then run a staging drill: stop the Redis and sentinel containers for 10 minutes under load.

## Risks and invariants
- **Banned, deleted or deactivated users, and bumped `tokenVersion`, must take effect immediately.**
  - Guard: the degraded path reads Postgres with a short TTL; evictions that fail during the outage are replayed or `user_cache:*` is flushed on recovery.
  - Tests: new `ResilientUserCacheUnitTest` (Redis throws → DB status used; ban during the outage is visible; pending evictions flushed on recovery). Existing `RedisUserCacheUnitTest`, `InMemoryUserCacheUnitTest`, `UserCacheServiceUnitTest`.
- **Database load.** Each authenticated request makes 2–4 cache reads (`isUserActive`, `getTokenVersion`, then `getAccountStatus` / `getEmailVerified` in the filters). The local TTL cache must absorb this. Check the prod Hikari pool size against peak requests per second.
- **Consent and email-verification enforcement must return the same answers when degraded.** Existing filter tests plus a degraded-mode case each for `ConsentEnforcementFilter` and `EmailVerificationEnforcementFilter`.
- **Infrastructure failure must never sign users out.** New `JwtAuthenticationFilter` unit test (cache throws `RedisConnectionFailureException` → 503 on private endpoints, anonymous on public ones), next to `JwtAuthenticationFilterBrokenSessionPublicUnitTest`. Frontend `error.interceptor` spec: 503 → no `session.clear()`.
- **Step-up brute-force protection must not weaken.** It fails closed. Test: `StepUpAuthService` with the store unavailable → 503 and no token issued.
- **Verification codes stay single use.** `getAndDelete` semantics in the in-memory store. Codes issued during the outage live only on that instance, which is acceptable only with one instance (HYPOTHESIS).
- **Rate limits while degraded are per instance and reset at the transitions.** Existing `RateLimiterServiceUnitTest`, `GdprRateLimiterUnitTest`, `rate-limiting.feature`; add a breaker-open test.
- **Upload quota drift.** Uploads recorded or deleted during the outage are not counted. Accept it and log it, or reconcile from the file tracking data (HYPOTHESIS that `FileTrackingService` can give per-user totals). Tests: `StorageRateLimitServiceUnitTest`, `UploadSystemHealthIndicatorUnitTest`.
- **GDPR.** In-memory fallbacks must keep TTLs and hashed keys. `InMemoryUserCache` currently logs the raw Firebase UID at INFO; switch it to `LogSafe` if it becomes a production path.
- **Health.** If a monitor relies on the overall health turning DOWN for Redis, it must switch to the new gauge or alert.
- **End to end.** The e2e profile uses a real Redis through Testcontainers. Add a Cucumber scenario that pauses the Redis container, then checks that login plus a private call still work and step-up returns 503. This assumes the harness exposes the container (HYPOTHESIS).

## Evidence
- FACT: `storage.mode` defaults to `redis` and is parsed in `backend/src/main/java/com/sm/instagram/platform/config/StorageModeConfiguration.java` (`@Value("${storage.mode:redis}")`, `fromString`).
- FACT: the `no-redis` profile sets `storage.mode: in-memory`, `spring.data.redis.enabled: false`, excludes the Redis auto-configurations, sets `cache.type: simple` and disables Redis health (`backend/src/main/resources/application-no-redis.yml`, lines 42–63, 86–87, 179–189).
- FACT: `RedisConfiguration` is conditional on `spring.data.redis.enabled` (default true). It uses `REJECT_COMMANDS` with auto-reconnect and builds a `RedisCacheManager` with no error handler (`config/RedisConfiguration.java`, lines 50, 196–202, 300–324).
- FACT: production uses sentinel, `timeout: 1000`, `storage.mode: redis`, a rate-limit breaker with `fallback-to-memory: true`, `redis.startup.test.enabled: true` / `fail-fast: true`, and `management.health.redis.enabled: true` (`application-prod.yml`, lines 98–123, 263–299, 341–343).
- FACT: `GdprCompliantRateLimiterService` picks its delegate once in the constructor from the storage mode (lines 65–84).
- FACT: `RedisRateLimiterService.checkLimit` catches any exception from the breaker call and uses `checkLimitInMemory`, or denies when fallback is off (lines 179–197).
- FACT: `RedisUserCache.isUserActive`, `getTokenVersion`, `getAccountStatus`, `getEmailVerified` and `evict` call `redisTemplate` with no try/catch; only `cacheUser` catches (`auth/cache/RedisUserCache.java`, lines 59–68, 78, 122, 145, 174, 203).
- FACT: `JwtAuthenticationFilter` calls `userCache.isUserActive`, and on any exception returns 401 `error.auth.invalid_token` for non-public endpoints or continues as anonymous for public ones (lines 196, 272–281).
- FACT: `/refresh-session` is not treated as public (`JwtAuthenticationFilter.java` line 464–467, Grep only).
- FACT: the frontend clears the session and redirects to sign-in when the refresh call returns 401; 5xx does not clear it (`frontend/src/app/core/interceptors/error.interceptor.ts`, lines 40, 113–130).
- INFERENCE: during the outage, signed-in users are signed out on their next private call, because the facts above chain together: cache throws → 401 → refresh → 401 → clear.
- FACT: `StepUpAuthService` and `EmailVerificationService` inject `StringRedisTemplate` with no condition and call it without handling Redis errors (`auth/stepup/StepUpAuthService.java`, lines 34, 101, 127–159, 210, 223, 237; `EmailVerificationService.java`, lines 127, 391–418).
- HYPOTHESIS: the `no-redis` profile cannot start, because no `StringRedisTemplate` bean exists when `RedisAutoConfiguration` is excluded. Only these two classes use that type (Grep). Check: `mvn spring-boot:run -Dspring-boot.run.profiles=no-redis`.
- FACT: `@Cacheable` is used on `city/CityRepository.java:14` and `auth/service/RecaptchaService.java:194`, and no `CacheErrorHandler` or `CachingConfigurer` exists in `src/main/java` (Grep).
- INFERENCE: both fail while Redis is down, because Spring's default cache error handler rethrows. Not read in code; confirm with a unit test.
- FACT: `RedisGeoLocationCache.get`/`put` catch and return null or log; `evict`, `tryLock` and `releaseLock` do not (lines 69–85, 102–109, 125, 229–238).
- FACT: `GeoLocationFacade` takes `cache.tryLock` before the database update (lines 143–153).
- FACT: `SessionSecurityService` has no Redis references (Grep). Its impossible-travel check waits 500 ms, lets the request through on timeout or error, and blocks high-privilege roles after the failure threshold (lines 277–304).
- FACT: `SocialAuthSessionService` stores sessions in a `ConcurrentHashMap` (lines 18–21).
- FACT: `StorageRateLimitService.checkAndRecordUpload` lets content uploads through on a Lua error (lines 470–475).
- FACT: the profile-photo path calls `hasStorageSpace` outside any try block, and `hasStorageSpace` reads Redis unwrapped (lines 374–385, 563–564).
- FACT: `recordUpload` and `decreaseUserStorage` rethrow as `StorageTranslatableException` (lines 329–332, 544–548).
- FACT: `UploadSystemHealthIndicator` returns DOWN when Redis is configured but its set/get probe fails (lines 93–96, 112–116, 140–155).
- FACT: the production app container's healthcheck calls `/api/actuator/health`, and the container uses `restart: unless-stopped` (`docker-compose-prod.yml.j2`, lines 226–232).
- FACT: `RedisStartupConnectivityTest` is enabled by `redis.startup.test.enabled` and decides fail-fast from `System.getProperty("spring.profiles.active")`. It never reads a `fail-fast` property (lines 27, 151–156).
- FACT: prod passes the profile as `SPRING_PROFILES_ACTIVE={{ spring_profile }}` and starts with `exec java ${JAVA_OPTS} -jar` (`docker-compose-prod.yml.j2:201`, `deployment/prod/spring-boot-entrypoint.sh:211`).
- HYPOTHESIS: `JAVA_OPTS` does not include `-Dspring.profiles.active`, so a restart during the outage continues without Redis instead of failing. Check the prod `.env` or secrets for `JAVA_OPTS`.
- FACT: `NoOpRedisHealthIndicator` uses `matchIfMissing = true` on `storage.mode=in-memory` (line 14).
- FACT: many user-mutation paths call `userCacheService.evict` right after `userRepository.save`: `UserService.java:879`, `LegalConsentService.java:422, 523`, `TwoFactorStatusController.java:352`, `EmailChangeService.java:110` and others (Grep).
- INFERENCE: those operations fail, and roll back where they run in a transaction, because `RedisUserCache.evict` throws. Check the `@Transactional` boundaries per method.
- HYPOTHESIS: `GeoLocationGdprService` is not active in prod, since it is conditional on `geoip.cache.type=redis` and that key did not appear in `application-prod.yml` (Grep). Check `/actuator/beans` in staging.
- HYPOTHESIS: production runs one backend instance (one app service in `docker-compose-prod.yml.j2`). This decides whether in-memory failover for verification codes is acceptable. Check the inventory or deployment with ops.
- HYPOTHESIS: `TokenExchangeService` (the login flow) also reads the cache in a way that throws, which would break login as well as later calls. Read its `userCacheService` calls before step 3.
