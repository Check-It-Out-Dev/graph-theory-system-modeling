I've read enough to answer. Nothing was modified.

# Redis outage in production: what breaks, how it's wired, and a runtime-degradation design

## Problem

In production, Redis is not a cache the app can do without. Today, losing it for an hour means:

- **Every signed-in user gets logged out.**
- **Nobody can sign in.**
- **Admin, consent and account-changing actions fail.**
- **Step-up codes, email verification and magic links fail.**
- **Rate limiting keeps working**, but only through a per-instance in-memory fallback that also floods the logs.

The only designed "no Redis" mode is the `no-redis` profile, chosen at startup. There is no runtime switch. The profile may not even start today, because two security services need a Redis template no matter which mode is active.

### Outage impact matrix (prod: `storage.mode=redis`, Sentinel)

| Area | During the outage | Class |
|---|---|---|
| Authenticated API calls | `JwtAuthenticationFilter` calls `userCache.isUserActive()`. `RedisUserCache` reads Redis with no try/catch, so the exception reaches the filter's catch-all, which returns **401 `error.auth.invalid_token`**. The frontend then calls `/auth/refresh-session`. That endpoint is not public, so it hits the same exception and also returns 401. The frontend treats that as a hard failure and **clears the session and redirects to sign-in**. | **FAILS**, with a mass logout |
| Login (`TokenExchangeService`) | `isUserActive` throws and the generic catch turns it into `error.auth.token_exchange_failed` | **FAILS** |
| Public / optional-auth endpoints | The filter drops a broken session to anonymous | **WORKS** |
| Banned / consent / email-verified filters | Would throw too, but the JWT filter fails first | **FAILS** |
| Writes that evict the user cache (admin status or role change, consent re-accept, email change, 2FA toggle, account deletion) | `RedisUserCache.evict` has no try/catch, so the exception reaches the business method | **FAILS** |
| Step-up auth (email code, TOTP token) | `StepUpAuthService` calls `StringRedisTemplate` directly | **FAILS** |
| Email verification links and magic-link / oobCode verify | `EmailVerificationService.storeOobCode/lookupOobCode` go straight to Redis | **FAILS** |
| Upload signed URLs | `StorageRateLimitService.hasStorageSpace` reads Redis outside a try block | **FAILS** |
| City lookups through `@Cacheable("citiesCache")` | `RedisCacheManager` with no `CacheErrorHandler` anywhere | **FAILS** |
| HTTP rate limiting | `RedisRateLimiterService` circuit breaker falls back to an in-memory window. Counters reset and are per instance. When a call fails (not when the breaker simply rejects it), there is a stack-trace log at ERROR plus one at WARN. Every 10 s the breaker lets up to 10 trial calls through to the dead Redis. | **DEGRADES** |
| Geo-IP lookup | Cache get/put swallow errors, so lookups go to the local MaxMind file. Every failed cache call logs a stack trace at ERROR. | **DEGRADES** |
| Impossible-travel check | 500 ms budget. If Redis calls are slow rather than refused, after 5 timeouts in a row ADMIN, PENDING_ADMIN and COMPANY sessions whose IP changed are rejected. | **DEGRADES**, possibly fails closed |
| Weekly MaxMind update | `tryLock` is outside any try block, so the update is skipped | **FAILS** (only matters on Wednesdays at 02:00) |
| Session storage | Sessions are HMAC-signed JWT cookies. Spring Session is not in the pom; `SocialAuthSessionService` is in-memory. Storage is fine, but validation depends on the user cache. | **WORKS**, but unusable because of the user cache |
| `/actuator/health` (overall) | Boot's Redis indicator and `UploadSystemHealthIndicator` both go DOWN, so the endpoint returns 503 | **FAILS** |
| `/actuator/health/readiness` and `/liveness` | Readiness checks db, diskSpace and liquibase; liveness checks ping. Neither includes Redis. | **WORKS** |
| Docker healthcheck | Hits the overall endpoint (60 s interval, 20 retries), so the container turns unhealthy after about 20 minutes. Plain Docker does not restart unhealthy containers. | Marked unhealthy, no automatic action |
| Startup during the outage | `docker compose up` waits for `redis: service_healthy`, so the app never starts. If Docker restarts the app on its own, both startup checks log and carry on (the fail-fast check reads a system property that isn't set), so it boots already broken. | Deploys **blocked**; restarts come up broken |

## Where it lives today

**Mode selection happens once, at startup.**
- `backend/src/main/java/com/sm/instagram/platform/config/StorageModeConfiguration.java` reads `storage.mode` (`redis` by default, or `in-memory`).
- Each implementation carries `@ConditionalOnProperty(storage.mode)`. So exactly one of each pair exists in the context:
  - `RedisUserCache` / `InMemoryUserCache` (`auth/cache/`)
  - `RedisRateLimiterService` / `InMemoryRateLimiterService` (`common/ratelimit/`)
  - `RedisGeoLocationCache` / `InMemoryGeoLocationCache` (`common/security/geoip/`)
  - `StorageRateLimitService` / `InMemoryStorageRateLimitService` (`storage/service/`)
- `GdprCompliantRateLimiterService` picks its delegate in its constructor. `UnifiedStorageConfiguration` only logs.
- `config/RedisConfiguration.java` is gated by `spring.data.redis.enabled`. It sets Lettuce to `REJECT_COMMANDS` with auto-reconnect and a pool with `testOnBorrow`. It builds `RedisCacheManager` without an error handler.

**The `no-redis` profile** (`src/main/resources/application-no-redis.yml`):
- sets `spring.data.redis.enabled=false`
- excludes `RedisAutoConfiguration`
- sets `spring.cache.type=simple` and `storage.mode=in-memory`
- disables the Redis health indicator.

However, `StepUpAuthService` and `EmailVerificationService` require a `StringRedisTemplate` unconditionally, and nothing else provides one. The profile most likely fails at startup. `dev-lite` uses a real Redis.

**Prod settings** (`application-prod.yml`):
- Sentinel mode, command timeout 1000 ms, connect timeout 5 s, pool max-wait 2000 ms
- rate-limit breaker threshold 3, open state 10 s, `fallback-to-memory: true`
- `redis.startup.test.enabled/fail-fast: true`
- `management.health.redis.enabled: true`

In `deployment/prod/docker-compose-prod.yml`, Redis runs with `allkeys-lru` and AOF. The app `depends_on` Redis and Sentinel being healthy, and its healthcheck calls `/api/actuator/health`.

**Flows that matter:**
1. **Request auth:** `JwtAuthenticationFilter` (lines 144–281) → `SessionSecurityService.validateSession` (geo) → `userCache.isUserActive` → `getTokenVersion` → the banned, consent and email-verification filters.
2. **Frontend recovery:** `frontend/src/app/core/interceptors/error.interceptor.ts`. A 401 or 419 triggers a refresh; a 401 from the refresh clears the session.
3. **Rate limiting:** `RateLimitInterceptor` → `GdprCompliantRateLimiterService` → `RedisRateLimiterService.checkLimit` (breaker, then in-memory fallback).
4. **Health and startup:** `config/RedisStartupConnectivityTest.java`, `common/redis/RedisValidationService.java`, `storage/health/UploadSystemHealthIndicator.java`, `common/validator/ApplicationStartupValidator.java`.

## Proposed change

**Replace the startup switch with one runtime availability signal, plus a deliberate policy for each capability.** Callers keep the interfaces they already use.

1. **`RedisAvailability`** (new, `common/redis/`)
   - A state machine: `UP → DEGRADED → RECOVERING → UP`.
   - It is fed by a scheduled PING (about every 2 s, short timeout) and by passive failures that a classifier recognises as connection problems. Those are Lettuce `RedisConnectionException`, `RedisCommandTimeoutException` and "not connected". Command errors such as WRONGTYPE do not count.
   - It moves to UP only after N probe successes in a row, so a flapping Redis doesn't bounce the state.
   - It exposes a Micrometer gauge and publishes Spring events.
   - It replaces the private circuit breaker inside `RedisRateLimiterService` with one shared breaker.
   - **While DEGRADED, the hot path makes no Redis calls at all.** That removes both the added latency and the log flood.

2. **User cache: go to the database, never "allow".**
   - A `@Primary ResilientUserCache` decorator wraps `RedisUserCache`.
   - When DEGRADED, or when a call fails, it loads a single snapshot (status, role, tokenVersion, emailVerified) from Postgres. A small local Caffeine cache (about 30 s TTL) keeps load reasonable: today one request can trigger up to 5 separate DB lookups on a miss.
   - `evict` never throws. It clears the local entry.
   - On recovery, it enters **quarantine** for the cache TTL (5 min). During quarantine, reads skip Redis but writes still go through. Any stale entry written before the outage, including one whose eviction failed, has expired by the end of quarantine. That keeps today's "a ban takes effect within 5 min" guarantee without scanning keys.

3. **Rate limiting: keep enforcing it.**
   - Use the existing in-memory window, switched by `RedisAvailability`.
   - Replace the unbounded `fallbackWindows` map with a size-limited Caffeine cache, so attackers can't grow it with endless IP or device keys.
   - Log state transitions once, not once per request.
   - Add an optional `instance-divisor` for the day prod runs more than one instance.

4. **Step-up, email-verification oobCodes and magic links: fail closed with a clear message.**
   - Put these behind an `EphemeralSecurityStore` interface.
   - When Redis is unavailable, return **503 + Retry-After** with a translated message.
   - There is no in-memory fallback in `redis` mode, because single-use tokens and lockout counters must never live in a store that disagrees with Redis. The in-memory implementation exists only for `storage.mode=in-memory`.
   - These flows are low-volume; an hour's delay is acceptable. A 500 or a logout is not.

5. **Spring cache:** add a `CachingConfigurer.errorHandler()`. A failed get counts as a miss; failed puts and evicts become logged no-ops.

6. **Geo-IP:** skip Redis while DEGRADED and use a small local cache. Guard `tryLock`, `releaseLock` and `evict`. A cache miss must never count toward the impossible-travel failure counter.

7. **Upload quotas:** wrap the Redis reads. While DEGRADED:
   - the file-size check still applies (it doesn't need Redis)
   - an in-memory global per-minute limit applies
   - per-user hourly and daily limits fail open
   - storage accounting is marked for reconciliation.

8. **Health:**
   - Remove Redis from the overall health status: set `management.health.redis.enabled=false`, and make `UploadSystemHealthIndicator` stop returning DOWN because of Redis.
   - Add a `redisAvailability` indicator that reports a custom `DEGRADED` status mapped to HTTP 200.
   - Point the Docker healthcheck at readiness.
   - Alert from the metric, not from container health.

9. **Startup:** prod starts in DEGRADED mode instead of blocking.
   - Replace `System.getProperty("spring.profiles.active")` with an explicit `redis.startup.required` property (default `false`), checked through `Environment`.
   - Change the compose dependency on Redis to `service_started`.

10. **`no-redis` becomes "permanently DEGRADED, no probe."** It uses the same decorators and the in-memory stores, so the code path used in local development is the one that runs during a real outage.

**Why this design:** the interfaces (`UserCacheService`, `GeoLocationCache`, `RateLimiterService`) already exist. Decorators keep call sites unchanged. And each capability's policy (use the DB, enforce locally, fail closed, fail open) is written down explicitly instead of being a side effect of whichever exception gets caught.

## Plan

1. **Availability core.**
   - New: `common/redis/RedisAvailability.java`, `RedisFailureClassifier.java`.
   - Change: `config/RedisConfiguration.java` (tighter prod pool max-wait and connect timeout for the request path; expose the shared breaker), `application.yml`, `application-prod.yml` (new `redis.resilience.*` keys).
2. **Resilient user cache.**
   - New: `auth/cache/ResilientUserCache.java` (`@Primary`, local cache, quarantine).
   - Change: `auth/cache/RedisUserCache.java` (no DB fallback inside it, so the decorator owns that decision).
   - Tests: new `ResilientUserCacheUnitTest`; update `unit/service/RedisUserCacheUnitTest.java`.
3. **Rate limiter.**
   - Change: `common/ratelimit/RedisRateLimiterService.java` (shared availability, bounded fallback, one log line per state change).
   - Tests: new degraded-mode unit test; keep `unit/service/GdprRateLimiterUnitTest.java` passing.
4. **Cache error handler.** New `config/CacheResilienceConfiguration.java`.
5. **Geo-IP.**
   - Change: `common/security/geoip/RedisGeoLocationCache.java`, `GeoLocationFacade.java` (`updateGeoIpDatabase`).
   - Tests: `unit/service/GeoIpServiceUnitTest.java`, `GeoLocationServiceUnitTest.java`.
6. **Ephemeral security store.**
   - New: `common/redis/EphemeralSecurityStore.java`, `RedisEphemeralSecurityStore.java`, `InMemoryEphemeralSecurityStore.java` (in-memory mode only), and a `ServiceTemporarilyUnavailableException` mapped to 503 in the existing `@ControllerAdvice`.
   - Change: `auth/stepup/StepUpAuthService.java`, `auth/service/EmailVerificationService.java`, `messages_*.properties`.
   - Frontend: map 503 in `shared/components/step-up-dialog/step-up-dialog.component.ts`, `feature/profile/email-change.component.ts`; add strings to `assets/i18n/en.json` and `pl.json`.
   - Tests: `unit/service/EmailVerificationServiceUnitTest.java`, a new `StepUpAuthServiceUnitTest`.
7. **Upload quotas.**
   - Change: `storage/service/StorageRateLimitService.java` (`hasStorageSpace`, `getGracePeriodMultiplier`, `recordUpload`).
   - Tests: `unit/service/StorageRateLimitServiceUnitTest.java`.
8. **Health.**
   - New: `health/RedisAvailabilityHealthIndicator.java`.
   - Change: `storage/health/UploadSystemHealthIndicator.java`, `application.yml` and `application-prod.yml` (`management.health.redis.enabled`, status mapping), `deployment/prod/docker-compose-prod.yml`, `deployment/test/docker-compose-test.yml`, `ansible/roles/11-docker-compose/templates/docker-compose-{prod,test}.yml.j2`.
   - Tests: `unit/service/StorageHealthUnitTest.java`.
9. **Startup.** Change `config/RedisStartupConnectivityTest.java` (use `Environment` and `redis.startup.required`) and `common/validator/ApplicationStartupValidator.java` (report the real state). Also, in `common/redis/RedisValidationService.java`, stop writing test and `rate_limit:*` keys in prod.
10. **`no-redis` profile.** Change `application-no-redis.yml` (pin `RedisAvailability` to DEGRADED) and `config/StorageModeConfiguration.java`. Add a context-load test for the `no-redis` profile.
11. **Chaos integration test.** New `integration/resilience/RedisOutageIntegrationTest.java`, built on `integration/config/ServiceIntegrationTestConfig.java`. It pauses the Testcontainers Redis, then checks:
    - login and an authenticated GET still work
    - 429s are still enforced
    - step-up returns 503
    - readiness stays UP
    - after unpausing, the quarantine is honoured.

    The test must turn the circuit breaker back on, because test properties disable it.

## Risks and invariants

| Invariant / risk | Guard |
|---|---|
| **I1.** A BANNED, DELETED or tokenVersion-bumped user never keeps access longer than today's 5-minute bound, including when an eviction fails during the outage and Redis then recovers. | New `ResilientUserCacheUnitTest` (evict during outage, recover, reads skip Redis for the TTL). Existing `unit/security/JwtAuthenticationFilterStaleTokenUnitTest.java`. The chaos IT. |
| **I2.** Rate limiting never silently allows everything: while degraded, a key still gets 429 at its limit. | New degraded rate-limiter unit test; the chaos IT. |
| **I3.** Step-up tokens stay single-use and lockout counters never come from a second store in `redis` mode. | Context test asserting no `InMemoryEphemeralSecurityStore` bean when `storage.mode=redis`; `StepUpAuthServiceUnitTest`. |
| **I4.** A cache failure never rolls back a database write (status change, consent, email change). | Unit tests where `evict` throws and the transaction still commits (`UserService`, `LegalConsentService`). |
| **I5.** Readiness and liveness never depend on Redis. | The existing groups (`application.yml:359-363`) plus an actuator IT. |
| **I6.** While DEGRADED, the hot path makes zero Redis calls. | Mock-verified "no interactions" tests on each decorator. |
| **I7.** A broken cookie on public endpoints still drops to anonymous. | Existing `unit/security/JwtAuthenticationFilterBrokenSessionPublicUnitTest.java`. |
| **Risk: Postgres load.** Every authenticated request reads the user from the DB. | The local cache limits this to one read per user per TTL. Hikari pool size must be load-tested (not read here). |
| **Risk: more than one instance.** In-memory limits multiply by the instance count. | `instance-divisor` setting; document it. |
| **Risk: flapping.** | Hysteresis (N successes to recover) and quarantine. |
| **Risk: fallback memory growth.** | Size-limited Caffeine caches. |
| **Pre-existing, not caused by this change:** `allkeys-lru` can evict quota and lockout keys; `RecaptchaService.isValidToken`'s `@Cacheable` has no callers. | Flag separately. |

## Evidence

- **FACT:** `RedisUserCache.isUserActive/getTokenVersion/getAccountStatus/getEmailVerified` call `redisTemplate.opsForValue().get` outside any try block, and `evict` has none either. Source: `auth/cache/RedisUserCache.java:78,122,145,174,203`.
- **FACT:** The JWT filter calls `isUserActive` (line 196) and `getTokenVersion` (219). Its catch-all returns 401 `error.auth.invalid_token` on private endpoints and anonymous on public ones (272–281). `/refresh-session` is not public (467–468). Source: `common/authorization/JwtAuthenticationFilter.java`.
- **FACT:** The frontend refreshes on 401/419 and clears the session and redirects when the refresh itself returns 401. Source: `frontend/src/app/core/interceptors/error.interceptor.ts:40,114-130`.
- **INFERENCE:** That combination logs out every signed-in user during the outage.
- **FACT:** Login calls `isUserActive` (`TokenExchangeService.java:204`) and wraps unknown exceptions as `token_exchange_failed` (399–401).
- **FACT:** `evict` is called inside write paths: `UserService.java:761,879`, `LegalConsentService.java:422,523`, `TwoFactorStatusController.java:352`, `EmailChangeService.java:110`.
- **INFERENCE:** Those operations fail, and roll back where they are transactional.
- **FACT:** `StepUpAuthService` and `EmailVerificationService` inject `StringRedisTemplate` unconditionally and use it without catching errors. Source: `StepUpAuthService.java:34,101,127,210`; `EmailVerificationService.java:51,394,405`.
- **INFERENCE:** The `no-redis` profile can't satisfy that dependency, because it excludes `RedisAutoConfiguration` (`application-no-redis.yml:58-63`). Not verified by running it.
- **FACT:** The rate limiter has a breaker (window 10, minimum calls equal to `failureThreshold`, 50 % failure rate) and falls back to an unbounded `ConcurrentHashMap`. Failed calls log with a stack trace at ERROR and again at WARN. Source: `RedisRateLimiterService.java:35,152-157,184-192,239`. Prod threshold 3 and open state 10 s: `application-prod.yml:278-282`.
- **INFERENCE:** Resilience4j's default of 10 permitted calls in half-open applies, since the code doesn't set it.
- **FACT:** Tests disable the breaker. Source: `src/test/resources/application-test.properties:156`, `application-integration.properties:161`.
- **FACT:** No `CacheErrorHandler` or `CachingConfigurer` exists (grep). `@Cacheable("citiesCache")` is at `city/CityRepository.java:14`. `RecaptchaService.isValidToken` has no callers (grep).
- **FACT:** `RedisGeoLocationCache` get/put catch errors and log a stack trace; `tryLock`, `releaseLock` and `evict` don't catch. Source: `RedisGeoLocationCache.java:81-85,107-109,229-239`. `GeoLocationFacade.updateGeoIpDatabase` calls `tryLock` before its try block (143).
- **FACT:** Impossible travel uses a 500 ms budget and, after the failure threshold, fails closed for ADMIN, PENDING_ADMIN and COMPANY. Source: `SessionSecurityService.java:277-315`.
- **HYPOTHESIS:** Redis calls during the outage are slow enough to hit that timeout. The answer depends on whether Lettuce rejects immediately or has to reconnect, which I haven't measured.
- **FACT:** `GeoLocationGdprService` needs `geoip.cache.type=redis`, which prod doesn't set, so it isn't loaded in prod (`GeoLocationGdprService.java:30`).
- **FACT:** `StorageRateLimitService.hasStorageSpace` reads Redis outside its try block (`StorageRateLimitService.java:564`). `SignedUrlService.java:164` calls it on every signed-URL request.
- **FACT:** Prod health config: Boot's Redis indicator is enabled (`application-prod.yml:341-343`), readiness covers db/diskSpace/liquibase and liveness covers ping (`application.yml:359-363`), and `UploadSystemHealthIndicator` reports DOWN when Redis is configured but unreachable (lines 93–96, 115).
- **FACT:** The compose healthcheck calls `/api/actuator/health` (60 s × 20 retries), the app `depends_on` Redis and Sentinel being healthy, and the restart policy is `unless-stopped`. Source: `docker-compose-prod.yml:285-304`.
- **INFERENCE:** Plain Docker doesn't restart containers just for being unhealthy.
- **FACT:** `RedisStartupConnectivityTest.isRedisRequired` reads `System.getProperty("spring.profiles.active")` (line 153). The profile is passed as the `SPRING_PROFILES_ACTIVE` env var (compose line 269), and the entrypoint runs `java ${JAVA_OPTS} -jar` with no `-D` for profiles. `RedisValidationService` catches everything and carries on (187–208). Nothing reads the `fail-fast: true` key in `application-prod.yml:299`.
- **INFERENCE:** Fail-fast never triggers in prod.
- **FACT:** `ApplicationStartupValidator` reports Redis mode as "operational" without probing (`ApplicationStartupValidator.java:137-138`).
- **FACT:** Sessions are HMAC-signed JWT cookies; the pom has no `spring-session`; `SocialAuthSessionService` is in-memory (its class comment, line 14).
- **FACT:** Redis runs with `allkeys-lru` and AOF (`docker-compose-prod.yml:97`). The user cache TTL is 5 minutes (`RedisUserCache.java:35`).
- **INFERENCE:** Because every user-cache key has a 5-minute TTL, a quarantine of the same length is enough to cover evictions that failed during the outage.
- **INFERENCE:** Prod runs a single app instance, since compose defines one `app` service. This matters for per-instance in-memory limits.
- **HYPOTHESIS:** Moving user lookups to Postgres (with a local cache) fits within the current Hikari pool. Prod pool size wasn't read; this needs a load test.
