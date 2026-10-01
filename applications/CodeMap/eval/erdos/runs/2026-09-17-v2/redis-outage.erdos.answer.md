# How checkItOut behaves during a one-hour Redis outage, and how to make it degrade gracefully

## Problem
If production Redis is down for an hour, you need to know what breaks, what gets worse and what keeps working, and how the `no-redis` fallback is wired. The goal is an architecture that detects the outage while the app is running and switches each feature to a defined fallback, instead of relying on a profile chosen at startup. Done means: a feature-by-feature outage table backed by code, a design, an ordered plan, and the rules the plan must keep with the tests that check them.

**The short answer:** production has no runtime fallback for Redis, apart from the API rate limiter and the geo-IP cache. The auth filter reads the user cache and doesn't handle a Redis error. So in effect **every signed-in feature fails with 401, and the frontend then signs users out**. Health checks keep reporting the instances as ready.

## Where it lives today

**Subsystems involved:** [3] rate limits and runtime config (rate limiter, geo-IP, health, `RedisConfiguration`, profiles), [6] user cache, [9] the auth filters and step-up login, [10] token exchange, and [170] the frontend's error interceptor.

### How the wiring works
- **One switch:** `storage.mode` (`redis` by default in `application.yml`; `in-memory` in `application-no-redis.yml`) picks implementations when the app starts. Each class carries its own `@ConditionalOnProperty`:
  - `RedisUserCache` or `InMemoryUserCache`
  - `RedisRateLimiterService` or `InMemoryRateLimiterService`
  - `RedisGeoLocationCache` or `InMemoryGeoLocationCache`
  - `StorageRateLimitService` or `InMemoryStorageRateLimitService`
  - `RedisValidationService` exists only in redis mode; `NoOpRedisHealthIndicator` only in in-memory mode.
- **`no-redis` profile:** it turns off `RedisConfiguration` (`spring.data.redis.enabled: false`), excludes Spring's Redis auto-configuration and uses the simple cache type.
- **Services that ignore the switch:** `StepUpAuthService` and `EmailVerificationService` always require a `StringRedisTemplate`. Nothing in `src/main` defines one, and the `no-redis` profile removes the auto-configuration that would. So the `no-redis` profile probably doesn't start at all (a hypothesis, see Evidence).
- **Settings:** prod uses Redis Sentinel, a 1 s command timeout, a 5 s connect timeout and a 2 s pool wait. The Lettuce client rejects commands while disconnected.

### What happens in a one-hour outage (prod, `storage.mode=redis`)

| Feature | Result | Why |
|---|---|---|
| Any signed-in request to a private endpoint | **FAILS: 401** | `JwtAuthenticationFilter` calls `userCache.isUserActive`. The Redis read in `RedisUserCache` isn't wrapped in try/catch, the filter's `catch (Exception)` turns the error into 401 `invalid_token`. |
| Frontend session | **FAILS: user is signed out** | On a 401, `error.interceptor.ts` calls `/auth/refresh-session`. That call also gets 401, which counts as a hard failure: session cleared, redirect to sign-in. |
| Sign-in, refreshing a session | **FAILS** | `TokenExchangeService` calls `isUserActive` (lines 204 and 1769) without handling errors. The exact HTTP status isn't checked. |
| Public endpoints for a signed-in visitor | **DEGRADES** | Same exception, but on public endpoints the filter treats the visitor as anonymous. |
| Public endpoints for anonymous visitors | **KEEP WORKING, but rate limits get looser and slower** | Details in the rate-limiting row. |
| Step-up codes/tokens, lockout, cooldown | **FAILS** | Every `StringRedisTemplate` call in `StepUpAuthService` is unguarded. |
| Sending and completing email verification | **FAILS** | `storeOobCode` and `lookupOobCode` are unguarded. |
| API rate limiting (`@RateLimit`) | **DEGRADES** | `RedisRateLimiterService` has a circuit breaker (prod: opens after 3 calls, stays open 10 s) with an in-memory fallback. But limits then apply per instance and counters start from zero. Until the breaker opens, and in each half-open retry, calls wait out the 1 s Redis timeout (inferred). |
| Upload pre-check (signed URL) | **FAILS** | The "fail open" in `checkUploadAllowed` only covers a missing Redis template. `hasStorageSpace` makes an unguarded Redis `get`. |
| Geo-IP / impossible-travel check | **DEGRADES** | `RedisGeoLocationCache.get` catches errors and returns a miss, so the local MaxMind lookup still runs. Latency can exceed `SessionSecurityService`'s 500 ms wait. After 5 failures in a row it **rejects sessions** for ADMIN, PENDING_ADMIN and COMPANY users whose IP changed. `tryLock` (weekly database update) is unguarded. |
| `@Cacheable` (`CityRepository.findByName`, `RecaptchaService.isValidToken`) | **FAILS** for city lookups | There's no `CacheErrorHandler`, so Redis errors reach the caller (Spring's default behaviour, not checked in code). The reCAPTCHA check itself is **unaffected**: the aspect calls `verifyToken`, which isn't cached. |
| Readiness (`/api/actuator/health/readiness`, the Docker HEALTHCHECK) | **Stays UP** | The readiness group is `db,diskSpace,liquibase`, so instances stay in rotation and give no signal. |
| Overall `/actuator/health` | **Shows DOWN** | `UploadSystemHealthIndicator` reports DOWN when Redis is configured but unreachable. |
| Startup validation if an instance restarts mid-outage | **Warns, but lies** | `RedisValidationService` logs "CONTINUING WITHOUT REDIS - FALLING BACK TO IN-MEMORY", but nothing actually switches. Prod turns on fail-fast, but `RedisStartupConnectivityTest.isRedisRequired()` reads the `spring.profiles.active` *system property*, while the image sets the profile via the `SPRING_PROFILES_ACTIVE` environment variable. So fail-fast probably never fires, and the instance boots into the same broken state. |
| Session cookies and fingerprint | **Keep working** | HMAC, JWT and fingerprint checks in `SessionSecurityService` don't use Redis. Only the geo part is affected. |

## Proposed change
**Design:** keep `storage.mode` for choosing the setup at deploy time (`redis` or `in-memory`). In redis mode, put one shared **`RedisAvailability`** component in front of Redis, and give each feature a **resilient wrapper** that tries Redis first and has a fallback chosen for that feature's security needs:

- **Shared outage detector:** `RedisAvailability` is one Resilience4j circuit breaker, already on the classpath. It counts only connection and timeout errors, runs a periodic PING while open, publishes metrics, and fires a "recovered" event. While it's open, callers skip Redis immediately, so no request waits on a timeout.
- **User cache:** use a new `ResilientUserCache` (marked `@Primary`). While Redis is out, it reads from PostgreSQL (the source of truth) through a local cache with a **30 s** lifetime. Evictions also clear the local cache. When Redis recovers, it replays the evictions it missed and clears the `user_cache:` keys using SCAN.
- **Spring cache:** a `CacheErrorHandler` treats cache errors as a miss.
- **API rate limiting:** keep the local fallback, but let the shared breaker decide when to use it. Optionally, for AUTH/STRICT limits, divide the limit by the configured number of instances.
- **Geo cache:** skip Redis while it's out and use a local in-memory cache. If the lock can't be taken, skip that run.
- **Step-up, email-verification codes, lockout counters:** these hold security state, so they should **fail closed** with a translatable 503 and `Retry-After`. The frontend already passes 5xx errors through without signing users out.
  - The alternative is a fallback table in PostgreSQL. It would keep working across instances, but adds a second store and tricky reads while Redis comes back, and an hour of "try again later" on step-up is an acceptable cost. I'd go with fail-closed now and revisit only if outages happen often.
  - These services would use a small `EphemeralStore` interface with Redis and in-memory versions, which also fixes startup under the `no-redis` profile.
- **Upload quotas:** handle errors. Use a conservative per-instance limiter, keep a local log of storage-usage changes, and replay it with `INCRBY` on recovery.
- **Health:** readiness stays independent of Redis, on purpose. Add a `redis` indicator that reports UP with `degraded=true` as a detail, plus alerting on the breaker state.

**Why this design:** the auth path can always fall back to PostgreSQL and stay correct. Only state that must be shared and used once (codes, lockouts) has no safe local copy, so that part fails closed.

## Plan
1. **Add detection and fix the misleading messages.** Nothing changes behaviour yet.
   - New `common/redis/RedisAvailability.java` and `RedisAvailabilityHealthIndicator.java`.
   - Fix the log message in `RedisValidationService.java`.
   - Replace the profile-name check in `RedisStartupConnectivityTest.java` with an explicit `redis.startup.required` setting (prod: `false` once step 3 ships).
   - Fix the `"inmemory"` typo in `ApplicationStartupValidator.java`.
   - Show Redis as a detail, not a failure, in `UploadSystemHealthIndicator.java`.
2. **Spring cache:** new `config/CacheErrorHandlerConfiguration.java` (implements `CachingConfigurer`).
3. **User cache:**
   - New `auth/cache/ResilientUserCache.java` wrapping `RedisUserCache` and a local cache.
   - Wire it in a new `config/ResilientStorageConfiguration.java`; build the local cache directly because `InMemoryUserCache` is conditional.
   - Keep the no-PII logging style of `RedisUserCache`.
   - Tests: extend `UserCacheServiceUnitTest`; add a `ResilientUserCacheUnitTest`.
4. **API rate limiter:** `RedisRateLimiterService.java` uses `RedisAvailability` instead of its private breaker. Optionally add a per-instance divisor in `RateLimitProperties.java`.
5. **Geo cache:** in `RedisGeoLocationCache.java`, skip Redis while it's unavailable, add a local cache, and handle `tryLock`/`releaseLock` errors. `GeoLocationFacade.java` then skips the update when the lock fails.
6. **Step-up and verification codes:**
   - New `common/ephemeral/EphemeralStore.java` with `RedisEphemeralStore` and `InMemoryEphemeralStore`.
   - Update `StepUpAuthService.java` and `EmailVerificationService.java`; `TestAuthController.java` stores codes too, so update it.
   - Add a 503 translatable exception and message keys in `messages_en/pl.properties`.
   - Frontend: show a "temporarily unavailable" message in the step-up and verification screens (`step-up.service.ts` and its callers).
7. **Upload quotas:** in `StorageRateLimitService.java`, handle errors in `hasStorageSpace`, `recordUpload`, `updateUserStorage` and `decreaseUserStorage`, and add the pending-changes log.
8. **Recovery:** a listener on the "recovered" event replays missed evictions and storage changes.
9. **Health config:** keep readiness as it is. Add the `redis` indicator to the overall health and to monitoring (`application-prod.yml`, `application-actuator.yml`), plus an alert on the breaker-state metric.
10. **Tests you can't skip:**
    - A Testcontainers integration test that pauses Redis during authenticated, step-up, upload and rate-limited calls, then un-pauses it and checks stale-entry cleanup.
    - A CI boot smoke test under `no-redis`.

## Risks and invariants
- **Ban and session revocation (`tokenVersion`, BANNED, deletion):** while Redis is down, a change can take up to 30 s to reach other instances (today's Redis lifetime is 5 min). Stale Redis entries must not survive recovery. Guarded by the step-3 unit tests, the pause/unpause integration test, and `security-advanced-session.feature`.
- **Step-up lockout and one-time codes must never be bypassed:** they fail closed and are never served from a per-instance store in prod. Guarded by `step-up-auth.feature` plus new 503 unit tests.
- **Load on PostgreSQL:** on a cache miss, each of `isUserActive`, `getTokenVersion`, `getAccountStatus` and `getEmailVerified` queries the database separately. The local cache has to absorb this, or the outage moves to the Hikari pool. Load-test with Redis paused.
- **Rate-limit accuracy:** fallback limits apply per instance, so with N instances the effective limit is N times higher. reCAPTCHA on login and signup reduces the brute-force risk. Guarded by `GdprRateLimiterUnitTest`, `RateLimiterServiceUnitTest` and `rate-limiting.feature`.
- **Quota drift:** storage-usage changes lost while Redis is down cause under-counting. Replaying the log on recovery fixes that, but if an instance restarts during the outage its log is lost. Reconciling from the upload tracking records would cover this (hypothesis). Guarded by `StorageRateLimitServiceUnitTest`.
- **Flapping:** half-open retries must allow only a few calls. Test the breaker's state changes.
- **GDPR:** local caches keep the hashed-key and no-PII-logging rules. `InMemoryUserCache` currently logs the Firebase UID, so don't reuse its log lines.
- **Existing tests to keep green:** `RedisUserCacheUnitTest`, `InMemoryUserCacheUnitTest`, `RedisServiceUnitTest`, `JwtAuthenticationFilterBrokenSessionPublicUnitTest`, `UploadSystemHealthIndicatorUnitTest`, `GeoIpServiceUnitTest`.

## Evidence
- FACT: `RedisUserCache.isUserActive`, `getTokenVersion`, `getEmailVerified` and `getAccountStatus` call `redisTemplate.opsForValue().get(key)` outside any try/catch. Only `cacheUser` catches errors. (backend/src/main/java/com/sm/instagram/platform/auth/cache/RedisUserCache.java:78, 145, 174, 203, 59-68)
- FACT: `JwtAuthenticationFilter` runs `validateSession` and then `userCache.isUserActive` inside a `catch (Exception)`. On non-public endpoints that catch writes 401 `error.auth.invalid_token`; on public ones it clears the context and continues. (backend/src/main/java/com/sm/instagram/platform/common/authorization/JwtAuthenticationFilter.java:176-204, 272-281)
- FACT: `TokenExchangeService` calls `isUserActive` without a guard during token exchange and session refresh. (backend/src/main/java/com/sm/instagram/platform/auth/service/TokenExchangeService.java:204, 1769)
- FACT: on 401/419 the frontend tries a refresh; if the refresh returns 401 it clears the session and redirects to sign-in, while 5xx errors leave the session alone. (frontend/src/app/core/interceptors/error.interceptor.ts:113-135)
- INFERENCE: signed-in users are signed out during the outage. The refresh call passes through the same filter, which 401s when the cache read throws, and `AuthController.refreshSession` also returns 401 when no authentication is set (backend/.../auth/AuthController.java:539-542).
- FACT: `RedisRateLimiterService` wraps Redis in a Resilience4j breaker and falls back to per-instance in-memory windows. Prod settings: failure threshold 3, open for 10 s, fallback to memory on. (backend/.../common/ratelimit/RedisRateLimiterService.java:151-196; backend/src/main/resources/application-prod.yml:269-282)
- FACT: `GdprCompliantRateLimiterService` chooses Redis or in-memory once, in its constructor, based on `storage.mode`. (backend/.../common/ratelimit/GdprCompliantRateLimiterService.java:65-84)
- FACT: `StepUpAuthService` uses `StringRedisTemplate` with no error handling for codes, attempts, cooldown, lockout and tokens. `EmailVerificationService.storeOobCode` and `lookupOobCode` are also unguarded. (backend/.../auth/stepup/StepUpAuthService.java:101-250; backend/.../auth/service/EmailVerificationService.java:391-418)
- HYPOTHESIS: the `no-redis` profile fails at startup because no `StringRedisTemplate` bean exists. A grep of `src/main` found no bean definition, and `application-no-redis.yml:58-63` excludes `RedisAutoConfiguration`. To check, run `mvn spring-boot:run -Dspring-boot.run.profiles=no-redis`.
- FACT: `RedisGeoLocationCache.get` and `put` catch errors, but `evict`, `tryLock` and `releaseLock` don't. `GeoLocationFacade.checkImpossibleTravel` recovers with `false` on error. (backend/.../common/security/geoip/RedisGeoLocationCache.java:69-85, 117-127, 225-239; GeoLocationFacade.java:101-119, 143-153)
- FACT: `SessionSecurityService` waits 500 ms for the travel check. After `circuit-breaker-threshold` (default 5) failures in a row, it returns "impossible travel" for ADMIN, PENDING_ADMIN and COMPANY. (backend/.../auth/service/SessionSecurityService.java:277-315)
- INFERENCE: during the outage Redis calls take close to the 1 s command timeout, based on `timeout: 1000` and the 2 s pool wait in application-prod.yml:112-122. That would cause the geo timeouts and the slow rate-limit calls before the breaker opens. How slow it really is depends on how Redis fails (connection refused versus network blackhole); measure it with the pause test.
- FACT: in `StorageRateLimitService`, "fail open" only applies when the template or properties are null. `hasStorageSpace` reads Redis unguarded, and `SignedUrlService` calls `checkUploadAllowed` and then `hasStorageSpace`. (backend/.../storage/service/StorageRateLimitService.java:193-197, 557-564; SignedUrlService.java:164-184)
- FACT: `@Cacheable` is used on `CityRepository.findByName` and `RecaptchaService.isValidToken`. No `CacheErrorHandler` or `CachingConfigurer` exists in `src/main`. `RecaptchaValidationAspect` calls `verifyToken`, not the cached method. (grep results; backend/.../auth/config/RecaptchaValidationAspect.java:66)
- INFERENCE: city lookups fail because Spring's default cache error handling rethrows. That is framework behaviour, not read here.
- FACT: the readiness group is `db,diskSpace,liquibase`, and the Docker HEALTHCHECK polls `/api/actuator/health/readiness`. (backend/src/main/resources/application.yml:359-363; backend/docker/instagram-platform/Dockerfile:158-159)
- FACT: `UploadSystemHealthIndicator` returns DOWN when Redis is configured and its set/get check fails. (backend/.../storage/health/UploadSystemHealthIndicator.java:93-115)
- FACT: `RedisValidationService` catches startup failures and logs a fallback that doesn't exist. `RedisStartupConnectivityTest.isRedisRequired` reads `System.getProperty("spring.profiles.active")`. (backend/.../common/redis/RedisValidationService.java:187-208; backend/.../config/RedisStartupConnectivityTest.java:151-156)
- INFERENCE: prod fail-fast doesn't fire. The runtime `JAVA_OPTS` don't set `-Dspring.profiles.active` (Dockerfile:162-185), and deployment sets the `SPRING_PROFILES_ACTIVE` environment variable (backend/.github/scripts/runner/generate-deployment-env.sh:55). To confirm, check the live JVM arguments on a prod host.
- FACT: `ApplicationStartupValidator` checks for `"inmemory"`, but the configured value is `in-memory`. (backend/.../common/validator/ApplicationStartupValidator.java:139; application-no-redis.yml:87)
- FACT: `InMemoryUserCache` has the same 5-minute lifetime and read-through-to-database behaviour as `RedisUserCache`, but it logs the Firebase UID. (backend/.../auth/cache/InMemoryUserCache.java:28, 50, 71-90)
- HYPOTHESIS: prod runs more than one backend instance, which is what makes per-instance fallbacks matter. Check `backend/src/main/resources/deployment-prod.yml`.
- HYPOTHESIS: `HealthController.java` (a custom `/health` endpoint) doesn't touch Redis. I didn't read it.
