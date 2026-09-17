## Problem
Production Redis is down for an hour (`storage.mode=redis`). I need to work out which features fail, which degrade and which keep working, and explain how the `no-redis` profile is wired today. Then I propose a runtime design: every Redis-backed feature has a defined fallback, pods stay ready and can start, and there is no stale security state once Redis is back. "Done" means a senior engineer can make the change step by step, and the system keeps working after each step.

## Where it lives today

**Subsystems:** [3] rate limits and runtime config (Redis config, rate limiter, storage limits, geo-IP, health); [6] user cache (`RedisUserCache`, `InMemoryUserCache`); [9] step-up auth, email verification and the post-auth filters; [10] token exchange; [11] `ConsentEnforcementFilter`. The frontend is unaffected apart from how it shows the errors below.

**How the switch works.** It is one startup-time property, `storage.mode` (default `redis` in `application.yml:242`, prod `application-prod.yml:264`). Every implementation picks itself with `@ConditionalOnProperty`:

| concern | `redis` bean | `in-memory` bean |
|---|---|---|
| user cache | `auth/cache/RedisUserCache.java` (`matchIfMissing=true`) | `auth/cache/InMemoryUserCache.java` |
| API rate limit | `common/ratelimit/RedisRateLimiterService.java` | `InMemoryRateLimiterService.java` |
| upload limits | `storage/service/StorageRateLimitService.java` | `InMemoryStorageRateLimitService.java` (extends it) |
| geo cache | `common/security/geoip/RedisGeoLocationCache.java` | `InMemoryGeoLocationCache.java` |
| health | Boot's `redis` indicator | `health/NoOpRedisHealthIndicator.java` |
| validation | `common/redis/RedisValidationService.java` | — |

- `GdprCompliantRateLimiterService` chooses its delegate once, in its constructor, from `StorageModeConfiguration`.
- `config/RedisConfiguration.java` is gated on `spring.data.redis.enabled`. It builds a Lettuce pool with `REJECT_COMMANDS` while disconnected, plus a `RedisCacheManager` for `@Cacheable`.
- `application-no-redis.yml` sets `storage.mode: in-memory` and `spring.data.redis.enabled: false`, excludes the Redis auto-configuration, uses `cache.type: simple` and turns off Redis health.
- Nothing switches at runtime. The only runtime fallback is inside the rate limiter.

**What an hour without Redis does in prod (redis mode):**

| feature | outcome | why |
|---|---|---|
| **Every authenticated API call** | **FAILS (401 `error.auth.invalid_token`)** | `JwtAuthenticationFilter:196` calls `userCache.isUserActive`. `RedisUserCache:78` calls `get()` outside any try, so the exception reaches the filter's `catch` at `:272`, which returns 401 on non-public endpoints. This happens even though a database fallback exists a few lines further down. |
| Login and token exchange | **FAILS** | `TokenExchangeService:204/1023/1143/1769` calls `isUserActive` and hits the same exception. |
| Public endpoints with a cookie | work, as anonymous | The filter's catch clears the security context when the endpoint is public or optional-auth. |
| Status changes (ban, email change, consent, deletion) | **at risk** | `evict()` at `RedisUserCache:122` has no try. Callers include `UserService`, `LegalConsentService`, `AdminCascadeDeleteServiceImpl`, `FirebaseAuthProxyService` and others. The write may fail or roll back, and an evict that never reaches Redis leaves stale `status` and `tokenVersion` for the cache TTL after Redis returns. |
| Step-up codes and tokens | **FAILS (500)** | `StepUpAuthService` calls `StringRedisTemplate` directly, with no catch around Redis (lines 101–244). |
| Email verification (oobCode) | **FAILS** | `EmailVerificationService:394/405/417` makes the same direct calls. |
| `@Cacheable` (`CityService`, `CityRepository`, `RecaptchaService`) | **FAILS** | `RedisCacheManager` is set up with no `CacheErrorHandler` anywhere in the codebase. |
| API rate limiting | DEGRADES to per-pod limits | `RedisRateLimiterService:180-193` uses a circuit breaker (50% failures over a window of 10 calls) and falls back to `checkLimitInMemory`. The effective limit becomes the configured limit × number of pods. |
| Upload limits | DEGRADE to fail-open | `StorageRateLimitService:470-474` returns `allowed(999,999)` on error, so hourly, daily and quota limits are not enforced. |
| Geo-IP and impossible travel | DEGRADE to cache misses | `RedisGeoLocationCache:81` and `:107` catch errors, so each lookup goes to MaxMind. |
| Social-auth session handoff | works | `SocialAuthSessionService` is a per-pod `ConcurrentHashMap` and never touched Redis. |
| Sessions (JWT cookies) | stateless, but see the first row | Only the active-user and `tokenVersion` checks touch Redis. |
| Health and readiness | **pods removed from service** if the actuator or monitoring profile is active | Readiness includes `redis` (`application-actuator.yml:26`, `application-monitoring.yml:35-38`). `UploadSystemHealthIndicator:93-115` also returns DOWN. |
| Startup and restarts during the outage | **crash loop** | `RedisStartupConnectivityTest:105-112` throws when the mode is redis and the `spring.profiles.active` system property contains `prod`. The Dockerfile passes it as `-D` (`docker/instagram-platform/Dockerfile:84`). The `redis.startup.test.fail-fast` yml key is never read. |

**Net result:** the API is effectively down for signed-in users for the whole hour, even though Postgres could answer every security check.

## Proposed change
Keep `storage.mode` as the *preferred* backend. Add a **runtime availability layer** so every Redis consumer has a chosen fallback.

1. **One `RedisAvailability` component** in [3]. It holds a shared resilience4j circuit breaker named `redis`, fed by command failures and a scheduled PING. It exposes `isAvailable()`, publishes `RedisDownEvent` and `RedisRecoveredEvent`, and emits a `redis.available` metric.
2. **Resilient decorators, not new profiles.** For each concern, a `@Primary` wrapper tries Redis through the breaker and otherwise falls back:
   - **User cache:** fall back to direct DB reads, with a small local TTL cache (about 30s) so the database isn't hit on every request. `evict` always clears the local entry and never throws. Rather than tracking every missed evict, **every user-cache key includes a generation number** that is bumped when Redis recovers. All entries written before or during the outage are then ignored, with no `KEYS *` scan. This keeps ban and `tokenVersion` changes correct.
   - **Step-up and oobCode:** fail closed with an explicit 503 `error.network.external_service` instead of a 500. They are security tokens that must be shared across pods, so an in-memory fallback would break multi-pod correctness. I chose this over a Postgres fallback store: an hour's outage of a rare flow is acceptable, and a JDBC store can be added later behind the same interface.
   - **Spring cache:** a `CacheErrorHandler` that logs the error and treats it as a cache miss.
   - **API rate limiter:** keep the existing fallback, but divide the limit by the expected replica count while in fallback.
   - **Upload limits:** fall back to `InMemoryStorageRateLimitService` logic, which is per-pod, instead of failing open.
3. **Health:** remove `redis` from the readiness group. Report Redis as a separate, non-aggregated `redisDegraded` status for alerting. Readiness means "can this pod serve", and it now can.
4. **Startup:** make `RedisStartupConnectivityTest` honour `redis.startup.test.fail-fast`, read through `Environment`, and set it to `false` in prod once steps 1–3 are live. The pod then starts in degraded mode.

The `no-redis` profile stays for development. It is simply the case where the breaker is permanently open.

## Plan
1. **Stop the 401 outage (smallest safe fix).** Wrap `get`, `set` and `delete` in `RedisUserCache` (`isUserActive`, `getTokenVersion`, `getAccountStatus`, `getEmailVerified`, `evict`, `evictAll`) so any failure falls back to `userRepository` and evict failures are logged. Files: `backend/.../auth/cache/RedisUserCache.java`, plus `RedisUserCacheUnitTest.java`.
2. **Add `CacheErrorHandler`.** Implement `CachingConfigurer.errorHandler()` in a new `config/ResilientCacheConfiguration.java`.
3. **Health and startup.** Remove `redis` from the readiness groups in `application-actuator.yml` and `application-monitoring.yml`. Take Redis out of the DOWN decision in `UploadSystemHealthIndicator` and report it as a detail. Make `RedisStartupConnectivityTest.isRedisRequired()` read `redis.startup.test.fail-fast` from `Environment`. Set the value in `application-prod.yml`.
4. **Availability layer.** Add `common/redis/RedisAvailability.java` (breaker, PING scheduler, events, metric) and a generation holder stored in Redis, cached locally and bumped on recovery. Move `RedisRateLimiterService`'s private breaker onto the shared one.
5. **Resilient user cache.** Add a new `auth/cache/ResilientUserCache.java` (`@Primary`, wraps `RedisUserCache` and uses a local TTL map). Change `HashingUtil.generateRedisKey` usage in `RedisUserCache` to include the generation. Handle `RedisRecoveredEvent` by bumping the generation.
6. **Fail closed, cleanly.** In `StepUpAuthService` and `EmailVerificationService`, check `RedisAvailability` and map `DataAccessException` to `NetworkTranslatableException("error.network.external_service")`, which returns 503. On the frontend, confirm the step-up and verification screens show the translated 503 message (`frontend` [170] `step-up.service.ts` and the error interceptor). Only add an i18n key if none is shown.
7. **Rate-limit degradation.** In the `RedisRateLimiterService` fallback, divide by `rate-limit.redis.fallback-replicas`. In `StorageRateLimitService`, replace the `catch` fail-open with a delegate to the in-memory logic (`InMemoryStorageRateLimitService`).
8. **Observability.** Add an alert on `redis.available == 0`, plus counters for fallback activations and 503s from step-up and verification.

Every step can ship on its own. Steps 1–3 already turn the outage into a degradation.

## Risks and invariants
- **Ban, deletion and `tokenVersion` must be honoured during and after the outage.** Ban, delete and tokenVersion checks read the database when Redis fails, and bumping the generation on recovery throws away stale entries. Tests:
  - `ResilientUserCacheUnitTest`: Redis throws → `isUserActive` reads the database.
  - Status is set to BANNED during the outage, Redis recovers, and the old cached entry still holds ACTIVE → the user is still reported BANNED.
  - `JwtAuthenticationFilter` test with a failing cache → 200 for an active user, 401 for a deleted one.
- **Evict must never roll back a business write.** Test: `UserService` ban with a throwing Redis template → the transaction commits.
- **Database load.** Every request falls back to `findByFirebaseUserId`, and the local 30s TTL limits that. Load-test with the Hikari pool size from prod.
- **Rate limiting is weaker.** Per-pod limits are an accepted degradation. Test: breaker open → the limit is divided by replicas.
- **Step-up must not be bypassable.** Fail closed. Test: Redis down → 503 and no action allowed.
- **Readiness change.** A pod that is truly unable to serve must still leave the pool; DB readiness stays in the group.
- **Lettuce latency before the breaker opens.** `REJECT_COMMANDS` fails fast once the connection is known to be down, but while TCP hangs each call waits for the command timeout. Keep `timeout` low in prod (the rate-limit `connection-timeout` is 1000 ms).
- **Integration test:** Testcontainers Redis stopped mid-test (existing `TestContainersConfig`) → authenticated GET succeeds, step-up returns 503, `/actuator/health/readiness` is UP. Restart Redis → the generation is bumped.

## Evidence
- FACT — the implementations are chosen only by `@ConditionalOnProperty(storage.mode)`; I read the annotations on all seven classes listed above.
- FACT — `application-no-redis.yml` sets `storage.mode: in-memory`, `spring.data.redis.enabled: false`, excludes the auto-configuration and disables Redis health (read in full).
- FACT — prod runs `storage.mode: redis` and `redis.startup.test.enabled: true` (`application-prod.yml:264,296-299`).
- FACT — `RedisUserCache.isUserActive` and `getTokenVersion` call Redis `get` outside a try (`:78`, `:145`); `evict` also has no try (`:122`).
- FACT — `JwtAuthenticationFilter:196-281` returns 401 from its catch on non-public endpoints.
- FACT — `RedisRateLimiterService:151-193` has a circuit breaker with in-memory fallback.
- FACT — `StorageRateLimitService:470-474` fails open.
- FACT — `RedisGeoLocationCache:69-85` swallows errors.
- FACT — `StepUpAuthService` and `EmailVerificationService` call `StringRedisTemplate` with no Redis catch (grep lines).
- FACT — there is no `CacheErrorHandler`, and `RedisCacheManager` is configured (`RedisConfiguration:300-323`, grep).
- FACT — readiness includes `redis` (`application-actuator.yml:26`, `application-monitoring.yml:35-38`).
- FACT — `UploadSystemHealthIndicator:93-115` returns DOWN when Redis fails.
- FACT — `RedisStartupConnectivityTest:105-112,151-156` throws based on the system property, and the Dockerfile sets `-Dspring.profiles.active`.
- FACT — `SocialAuthSessionService` is an in-memory `ConcurrentHashMap`.
- FACT — the Lettuce client uses `DisconnectedBehavior.REJECT_COMMANDS` (`RedisConfiguration:200`).
- INFERENCE — login fails because `TokenExchangeService` calls the same throwing `isUserActive` (call sites read via grep, not the surrounding catch).
- INFERENCE — `BannedUserAuthorizationFilter`, `EmailVerificationEnforcementFilter` and `ConsentEnforcementFilter` never reach Redis for non-public calls, because the JWT filter has already returned 401.
- HYPOTHESIS — an evict failure inside a transaction rolls back status changes.
- HYPOTHESIS — the prod deployment activates the actuator or monitoring profile, so the readiness group applies.
- HYPOTHESIS — the `no-redis` profile can start even though `StepUpAuthService` requires a `StringRedisTemplate`.

## Corrections after verification

**Corrections**

1. **Startup does not crash-loop in prod.** The prod stack is Docker Compose (`backend/deployment/prod/docker-compose-prod.yml`), not Kubernetes. It replaces the Dockerfile entrypoint with `spring-boot-entrypoint.sh`, which runs `exec java ${JAVA_OPTS} -jar /app/app.jar` (line 211). The profile is set only through the `SPRING_PROFILES_ACTIVE` environment variable (compose line 269) and is not passed as `-D`. `RedisStartupConnectivityTest.isRedisRequired()` reads `System.getProperty("spring.profiles.active")`, which is empty in this setup, so it only logs a warning. The crash-loop happens only on the Dockerfile's own start path (`Dockerfile:84`).
   - What really blocks a restart during the outage is `depends_on: redis: condition: service_healthy` (compose lines 292–295). `docker compose up` will not start the app while Redis is unhealthy.
   - Plan step 3 still stands: make the fail-fast key real, so the behaviour does not depend on how the JVM was launched. It also gains a deployment change: relax `depends_on` for `redis` and `redis-sentinel` to `service_started`.
2. **Health in prod.** Prod does not turn on the `actuator` or `monitoring` profile, so the `readiness` group that includes `redis` does not apply. The compose healthcheck calls the aggregate `/api/actuator/health` (line 300). With `management.health.redis.enabled: true` and `UploadSystemHealthIndicator`, that returns DOWN, and the container becomes `unhealthy` after the retries. Compose does not restart or remove unhealthy containers without an autoheal tool, and I found none. So today the result is a false "unhealthy" alarm, not lost traffic.
   - Plan step 3 changes: point the compose healthcheck at a group that leaves Redis out (for example `/actuator/health/readiness` with `include: db,diskSpace`), still remove `redis` from the readiness groups in `application-actuator.yml` and `application-monitoring.yml`, and report Redis as a detail rather than a DOWN.
3. **Login fails with a misleading error.** Confirmed: `TokenExchangeService:399-401` catches the Redis exception and throws `AuthenticationTranslatableException("error.auth.token_exchange_failed")`. Users see "token exchange failed", not a 5xx.
4. **Evict can roll back a status change (upgraded from HYPOTHESIS to INFERENCE).** `UserService:751-761` bumps `tokenVersion` and then calls `userCacheService.evict(...)` before commit, as the code comment says. An unchecked Redis exception there escapes the status-change method, so the ban or status change is lost. I did not read the `@Transactional` annotation itself. The plan-step-1 test "ban with Redis down commits" is the guard.
5. **Still unverified, but the plan does not depend on it:** whether the `no-redis` profile can start while `StepUpAuthService` and `EmailVerificationService` require a `StringRedisTemplate`. Among the main sources, only `StepUpAuthService` and `EmailVerificationService` reference `StringRedisTemplate`, so it may be failing today. Settle it by starting the app with `--spring.profiles.active=no-redis`. If it fails, route those two services through the availability layer from step 6.

Everything else in the answer stands. The headline result is unchanged: signed-in users get 401 on every protected call and cannot log in for the whole hour, even though Postgres could answer those checks. Plan step 1 alone turns that outage into a degradation.
