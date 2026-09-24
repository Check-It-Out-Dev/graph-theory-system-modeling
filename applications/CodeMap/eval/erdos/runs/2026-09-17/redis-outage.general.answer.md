I've read enough to answer. Below is the analysis, then the verification pass.

# Problem

Production runs `storage.mode=redis` with Sentinel. If Redis is gone for an hour, what breaks, what degrades and what keeps working? And how do we move from "choose Redis or in-memory by profile at startup" to "degrade at runtime"?

**Short answer.** Today a Redis outage logs out every signed-in user and blocks new logins. It also breaks uploads, step-up auth, email verification and some city lookups, and it turns `/actuator/health` DOWN. The only thing with a real runtime fallback is the HTTP rate limiter. The `no-redis` profile is a startup-time swap covering some components, and by my reading it doesn't even boot any more.

## Impact matrix (Redis down, prod profile)

| Feature | Outcome | Why |
|---|---|---|
| **Authenticated API requests** | **FAIL (401 → FE logout)** | `JwtAuthenticationFilter` calls `userCache.isUserActive()`. `RedisUserCache` runs `opsForValue().get()` outside any try, so it throws. The filter's generic `catch (Exception)` then writes **401 `invalid_token`**. The frontend calls `/auth/refresh-session`, which isn't public, so it gets 401 the same way. The FE treats a 401 from refresh as a hard failure: `session.clear()` and redirect to sign-in. |
| **Login / token exchange / refresh** | **FAIL** | `TokenExchangeService` calls `isUserActive` on the login and refresh paths (lines 204, 1023, 1143, 1769), and it throws. |
| Public / optional-auth endpoints | Keep working (as anonymous) | The filter catch clears the context and continues when `isPublic` or optional auth applies. |
| Admin status changes (ban, role change, archive, delete, email verify) | **FAIL (rollback)** | `userCacheService.evict()` has no guard and runs inside `@Transactional` methods (`UserService` 322/375/761/879, `UserAccountOrchestrator` 84, `FirebaseAuthProxyService.completeVerification` 1257, which rolls back and compensates Firebase). |
| HTTP rate limiting (`@RateLimit`) | **DEGRADES** (per-instance memory) | `RedisRateLimiterService` has a Resilience4j breaker plus `fallbackToMemory`, and prod sets it on. It's never reached for authenticated traffic, though, because the filter fails first. |
| Upload rate limit / quota | **FAIL** | `hasStorageSpace()` and `getGracePeriodMultiplier()` call Redis outside try, so `SignedUrlService.checkUploadAllowed` throws. The "fail open" branches only cover `redisTemplate == null` or the Lua script's own try. |
| Step-up auth (email code, admin TOTP token) | **FAIL** | `StepUpAuthService` uses `StringRedisTemplate` directly with no handling. |
| Email verification send / complete | **FAIL** | `storeOobCode` and `lookupOobCode` go straight to Redis. |
| `@Cacheable` (`citiesCache` via `CityRepository.findByName`) | **FAIL** (likely) | `RedisCacheManager` with no `CacheErrorHandler` anywhere, so city conversion in profile and opportunity mapping throws. |
| Geo-IP lookup | **DEGRADES** | `RedisGeoLocationCache.get`/`put` catch errors and fall through to MaxMind. The weekly DB-update lock (`tryLock`/`releaseLock`) and `evict` aren't guarded. |
| Impossible-travel check | **DEGRADES, can lock out privileged roles** | 500 ms budget. If Redis GETs are slow, the check times out repeatedly, and after 5 consecutive failures it fails **closed** for ADMIN, PENDING_ADMIN and COMPANY. |
| Sessions | Keep working (structurally) | Sessions are HMAC-signed JWT cookies plus `SessionSecurityService`, with no Spring Session. `SocialAuthSessionService` is a `ConcurrentHashMap`. They're still unusable in practice because of row 1. |
| Health `/api/actuator/health` | **DOWN (503)** | Boot's `RedisHealthIndicator` (`management.health.redis.enabled: true`) and `UploadSystemHealthIndicator` both go DOWN. The Docker healthcheck hits the aggregate endpoint, so the container is marked unhealthy. Nothing in the repo auto-restarts unhealthy containers. |
| Startup validation | **Does not fail fast (despite config)** | `RedisStartupConnectivityTest.isRedisRequired()` reads `System.getProperty("spring.profiles.active")`, but prod sets the `SPRING_PROFILES_ACTIVE` **env var**, so the check returns false and the app starts. `RedisValidationService` logs "FALLING BACK TO IN-MEMORY", but nothing actually falls back. `redis.startup.test.fail-fast` is never read. Compose `depends_on: redis: service_healthy` still stops `docker compose up` from starting the app. |
| Recovery | Automatic | Lettuce `autoReconnect(true)`. Stale `user_cache:*` entries whose evict failed during a short blip could let a just-banned user through for up to 5 minutes (TTL). |

# Where it lives today

**How the mode is chosen: one property, read once at startup**
- `backend/src/main/java/com/sm/instagram/platform/config/StorageModeConfiguration.java`: parses `storage.mode` (defaults to REDIS). It only logs and answers `isRedisMode()`.
- `config/UnifiedStorageConfiguration.java`: only logs. The comments say it "controls session management / upload tracking", which isn't true.
- `config/RedisConfiguration.java`: `@ConditionalOnProperty(spring.data.redis.enabled, matchIfMissing=true)`. It sets up Lettuce with `REJECT_COMMANDS`, `autoReconnect`, a command timeout and a pool with `testOnBorrow`. It also defines `RedisTemplate<String,Object>` (`@Primary`), a `RedisCacheManager` and `@EnableCaching`, the only one in the codebase.
- `resources/application-no-redis.yml`: sets `storage.mode: in-memory`, `spring.data.redis.enabled: false`, excludes `RedisAutoConfiguration`, sets `management.health.redis.enabled: false` and `cache.type: simple`.
- `resources/application-prod.yml`: sets `storage.mode: redis`, Sentinel mode, `timeout: 1000`, and `rate-limit.redis.circuit-breaker.*` plus `redis.startup.test.*`.

**Implementation pairs, each chosen with `@ConditionalOnProperty(storage.mode)`**

| Port | Redis impl | In-memory impl | Notes |
|---|---|---|---|
| `auth/cache/UserCacheService` | `RedisUserCache` (`matchIfMissing=true`) | `InMemoryUserCache` | No runtime fallback. |
| `common/ratelimit/RateLimiterService` | `RedisRateLimiterService` (no `matchIfMissing`) | `InMemoryRateLimiterService` | Picked in the `GdprCompliantRateLimiterService` constructor; this pair has the only runtime fallback. |
| `common/security/geoip/GeoLocationCache` | `RedisGeoLocationCache` | `InMemoryGeoLocationCache` | Used by `GeoLocationFacade`. |
| `storage/service/StorageRateLimitService` | concrete class (`matchIfMissing=true`) | `InMemoryStorageRateLimitService` | Plus `StorageConfiguration.fallbackStorageRateLimitService` (`@ConditionalOnMissingBean`). |
| Health | Boot `RedisHealthIndicator` | `health/NoOpRedisHealthIndicator` (`in-memory`, `matchIfMissing=true`) | |
| — | `common/security/GeoLocationGdprService` | — | Keyed on `geoip.cache.type=redis`, which no yml sets, so the bean and `GeoLocationGdprController` are never created. |

**Redis users with no in-memory counterpart:** `auth/stepup/StepUpAuthService`, `auth/service/EmailVerificationService` (both inject `StringRedisTemplate`), `@Cacheable` in `city/CityRepository` and `auth/service/RecaptchaService`, and `storage/health/UploadSystemHealthIndicator`.

**Flows that matter**
1. **Request path:** filters `JwtAuthenticationFilter` (session fingerprint → geo travel check → `isUserActive` → `getTokenVersion`) → `BannedUserAuthorizationFilter` / `ConsentEnforcementFilter` (`getAccountStatus`) → `EmailVerificationEnforcementFilter` (`getEmailVerified`) → MVC `RateLimitInterceptor`. That's four cache reads per authenticated request, all before the rate limiter.
2. **Frontend:** `frontend/src/app/core/interceptors/error.interceptor.ts` treats 401/419 as "refresh, then retry", and a 401 from refresh as "clear session and go to sign-in". Other statuses (5xx) leave the session alone.
3. **Startup:** `RedisValidationService` (`@PostConstruct`, logs only), `RedisStartupConnectivityTest` (`ApplicationRunner`, fail-fast is dead in prod), `ApplicationStartupValidator.checkStorageMode` (string only). Deploy is `deployment/prod/docker-compose-prod.yml` with a single `app` service.

# Proposed change

**Design: a shared Redis availability breaker, plus a resilient adapter per port with an explicit per-port fallback policy. The profile only picks the starting state.**

1. **`RedisAvailability` (new, `common/redis/`).** One Resilience4j `CircuitBreaker("redis")` shared by every adapter, fed by real call outcomes. A scheduled PING probe (every 2 s, 250 ms timeout) drives half-open → closed. It publishes `RedisUnavailableEvent` and `RedisRecoveredEvent`, exposes a Micrometer gauge, and is the single source for health.
   - **Why:** today each component finds the outage by timing out on its own. A shared open breaker means requests stop paying the Redis timeout; that matters for the 500 ms impossible-travel budget and the filter chain.
2. **One `@Primary Resilient*` adapter per port, wired in redis mode.** Each adapter holds both the Redis impl and a local fallback, both always built. When the breaker is closed it calls Redis, records any failure and falls back; when it's open it goes straight to the fallback. Fallback policy is chosen by how risky the data is:
   - **UserCache:** fall back to **DB-direct with a short local cache** (Caffeine, about 30 s TTL, request-scoped memo so the four reads per request become one query). The DB is the source of truth, so auth keeps working.
     - Failed evicts **must not throw into business transactions**. Record the UID in a bounded "dirty" set and evict locally.
     - On `RedisRecoveredEvent`, before closing the breaker, delete the dirty keys and SCAN-delete `user_cache:*` (5-minute TTL, so the keyspace is small). This keeps the invariant that "a ban takes effect immediately" across recovery.
   - **HTTP rate limiter:** keep the existing memory fallback, but move it onto the shared breaker. Per-instance counters are acceptable; prod runs a single app container.
   - **Upload limits:** fall back to `InMemoryStorageRateLimitService`-style counters for hourly, daily and global limits. The file-size limit stays enforced. The storage quota **fails open** with a metric, since usage is unknowable while Redis is down.
   - **Geo cache:** a breaker-open read is a cache miss (MaxMind lookup). `tryLock` falls back to a local lock, which is safe on a single instance. `evict` and `releaseLock` become best-effort.
   - **Step-up auth and email-verification oobCodes: fail closed** with a translated **503** (`error.service.temporarily_unavailable`), *not* a local store. Moving attempt counters and lockouts into local memory would silently reset brute-force lockouts set before the outage. These flows are rare and can wait an hour. A 503 doesn't trigger FE logout.
   - **Spring Cache:** add a `CachingConfigurer` with a `CacheErrorHandler` that logs and treats errors as misses.
3. **`JwtAuthenticationFilter` must not turn infrastructure errors into 401.** With the adapter in place the cache won't throw. As defence in depth, catch `DataAccessException`/`RedisConnectionFailureException` separately and return **503**, never 401. 401 means "your credentials are bad" to the FE, and misusing it is what causes the mass logout.
4. **Health: degraded ≠ down.** Replace the Boot redis indicator with one backed by `RedisAvailability` that reports custom status `DEGRADED`, mapped to HTTP 200 via `management.endpoint.health.status.http-mapping`. `UploadSystemHealthIndicator` stops going DOWN over Redis. Point the Docker healthcheck at `/api/actuator/health/liveness`, which already has a group in `application.yml`.
5. **Startup: start degraded instead of pretending to fail fast.** Replace `isRedisRequired()` with an explicit `redis.startup.required` property (default false). If it's false, the breaker starts OPEN and the probe closes it. Delete the false "FALLING BACK" log. Keep compose `depends_on` for normal deploys.
6. **`no-redis` becomes the same architecture with the breaker forced open** and no Redis beans. `StepUpAuthService` and `EmailVerificationService` depend on the new ports `StepUpStore` and `OobCodeStore` instead of `StringRedisTemplate`, so the profile boots, and the in-memory stores are fine for local dev.

# Plan

1. **Pin down current behaviour with failing tests first**, without changing prod code:
   - A unit test where `RedisUserCache` throws and `JwtAuthenticationFilter` returns 401 (the bug to flip).
   - A `RedisRateLimiterService` breaker-fallback test (none exists).
   - A context test that the `no-redis` profile starts.
   - Files: `src/test/java/.../unit/security/JwtAuthenticationFilterRedisOutageUnitTest.java` (new), `.../unit/service/RedisRateLimiterServiceUnitTest.java` (new), `.../integration/config/NoRedisProfileContextIT.java` (new).
2. **Add `RedisAvailability`:** breaker, probe, events, metrics. Files: `common/redis/RedisAvailability.java` (new), `common/redis/RedisAvailabilityEvents.java` (new), `resources/application.yml` and `application-prod.yml` (`redis.availability.*`; remove the dead `redis.startup.test.fail-fast`).
3. **User cache adapter:** add `auth/cache/ResilientUserCache.java` (new, `@Primary`, dirty set and recovery purge). Remove `@Component` selection from `RedisUserCache` and `InMemoryUserCache` and wire them from `config/UnifiedStorageConfiguration.java`. Add the `dataAccess → 503` catch in `common/authorization/JwtAuthenticationFilter.java`.
4. **Rate limiter onto the shared breaker:** `common/ratelimit/RedisRateLimiterService.java` (inject `RedisAvailability`, drop the private registry), `common/ratelimit/GdprCompliantRateLimiterService.java`.
5. **Upload limits:** `storage/service/StorageRateLimitService.java` (wrap the Redis reads in `hasStorageSpace`, `getGracePeriodMultiplier`, `updateUserStorage`, `getUserStatus`), plus `storage/service/ResilientStorageRateLimitService.java` (new) or a delegate to `InMemoryStorageRateLimitService.java`. Also check its 2-arg vs 3-arg `checkUploadAllowed` mismatch.
6. **Geo:** `common/security/geoip/RedisGeoLocationCache.java` (guard `evict`/`tryLock`/`releaseLock`, short-circuit on open breaker). Decide what to do with the dead `GeoLocationGdprService` condition (`geoip.cache.type` → `storage.mode`) as a separate ticket.
7. **Step-up and email verification ports (fail closed):** `auth/stepup/StepUpStore.java` + `RedisStepUpStore` + `InMemoryStepUpStore` (new), `auth/stepup/StepUpAuthService.java`, `auth/service/OobCodeStore.java` + impls (new), `auth/service/EmailVerificationService.java`, `common/exceptions/ServiceUnavailableTranslatableException.java` (new) plus a handler mapping, and `messages*.properties` (new i18n key).
8. **Spring Cache error handler:** `config/CacheResilienceConfiguration.java` (new, `CachingConfigurer`).
9. **Health:** `health/RedisAvailabilityHealthIndicator.java` (new; replaces `NoOpRedisHealthIndicator`), `storage/health/UploadSystemHealthIndicator.java`, `application-prod.yml` (`management.health.redis.enabled: false`, status order and http-mapping for DEGRADED), and `deployment/prod/docker-compose-prod.yml` plus `ansible/roles/11-docker-compose/templates/docker-compose-prod.yml.j2` (healthcheck → `/api/actuator/health/liveness`). The test compose and template get the same change.
10. **Startup:** `config/RedisStartupConnectivityTest.java` (property-driven `required`, seeds the breaker state), `common/redis/RedisValidationService.java` (remove the false fallback claim; skip the data-type write tests in prod), `common/validator/ApplicationStartupValidator.java` (`"inmemory"` → `"in-memory"`, report DEGRADED).
11. **`no-redis` profile:** `application-no-redis.yml` (`redis.availability.forced-open: true`), `config/StorageModeConfiguration.java`, `CONTRIBUTING.md`, `docs/DEV-LITE.md`.
12. **Outage integration test:** pause the Testcontainers Redis mid-test (`docker pause`/unpause) and assert that login, an authenticated GET, a ban taking effect after recovery, a step-up 503 and health 200 DEGRADED all behave. Files: `src/test/java/.../integration/resilience/RedisOutageIT.java` (new), reusing `e2e/config/TestContainersConfig.java`.
13. **Frontend (small):** make sure a 503 from step-up and verify-email shows a "temporarily unavailable" toast. Files: `frontend/src/app/core/interceptors/error.interceptor.ts` (no change expected), the step-up and verify-email components, and i18n.

# Risks and invariants

| Invariant | Risk | Guarding tests |
|---|---|---|
| A ban, role change or deactivation takes effect immediately | Local fallback cache or a failed evict serves stale "active" data after recovery | New `ResilientUserCacheUnitTest` (dirty set purged on recovery; local TTL ≤ 30 s); `RedisOutageIT` "ban during outage"; existing `security-advanced-session.feature`, `admin-user-management.feature` |
| 419 stale-token semantics unchanged | Adapter reorders `getTokenVersion` or skips it | Existing `JwtAuthenticationFilterStaleTokenUnitTest`, `JwtAuthenticationFilterBrokenSessionPublicUnitTest`, FE `error.interceptor.spec.ts` |
| Infra errors never produce 401 | Mass logout comes back | New `JwtAuthenticationFilterRedisOutageUnitTest` |
| Step-up brute-force limits can't be reset by an outage | Someone "helpfully" adds an in-memory fallback | New `StepUpAuthServiceOutageUnitTest` (expects 503); existing `RunStepUpAuthIT` / `step-up-auth.feature` |
| Rate limits still enforced (per instance) during an outage | Breaker misconfigured → fail-open | New `RedisRateLimiterServiceUnitTest`; existing `GdprRateLimiterUnitTest`, `RateLimitInterceptorUnitTest`, `rate-limiting.feature` |
| DB load under fallback | 4 cache reads → 4 queries per request for an hour | Request-scoped memo plus Caffeine; `UserCacheServiceUnitTest`; watch Hikari metrics (prod pool) |
| Multi-instance correctness | Local fallbacks are per instance; fine for today's single `app` container, wrong if scaled out | Document in `RedisAvailability` javadoc; revisit when adding replicas |
| Health semantics | DEGRADED mapped to 200 hides a real outage from monitoring | Alert on the `redis_availability` gauge/breaker state, not on HTTP status; `UploadSystemHealthIndicatorUnitTest` updated |
| `no-redis` boots | Regressions in bean wiring | New `NoRedisProfileContextIT` |
| Transaction rollback on evict | Evict failures currently roll back status changes | New `UserServiceEvictFailureUnitTest` |

# Evidence

1. FACT: `RedisUserCache.isUserActive`, `getTokenVersion`, `getEmailVerified` and `getAccountStatus` call `redisTemplate.opsForValue().get` outside try; `cacheUser` catches. Source: `auth/cache/RedisUserCache.java:78,145,174,203` and `59-68`.
2. FACT: `JwtAuthenticationFilter` calls `isUserActive` at step 5 inside a try whose `catch (Exception)` writes 401 `error.auth.invalid_token` unless the endpoint is public or optional-auth. Source: `common/authorization/JwtAuthenticationFilter.java:196,272-281`.
3. FACT: `/refresh-session` isn't public in the filter, and `createRefreshedSession` calls `isUserActive`. Source: `JwtAuthenticationFilter.java:467-468`; `TokenExchangeService.java:1769`.
4. FACT: the FE refreshes on 401/419 and on a refresh 401 runs `session.clear()` and redirects. Source: `frontend/src/app/core/interceptors/error.interceptor.ts:40,113-127`.
5. INFERENCE: Redis operations throw (connection failure or timeout) while Redis is unavailable. Source: Lettuce/Spring Data Redis behaviour plus `REJECT_COMMANDS` in `RedisConfiguration.java:200`.
6. FACT: `RedisRateLimiterService` has a Resilience4j breaker and an in-memory fallback, and prod enables `fallback-to-memory`. Source: `RedisRateLimiterService.java:151-197`; `application-prod.yml:278-282`.
7. FACT: no test targets `RedisRateLimiterService`'s fallback. Source: grep of `src/test` finds it only in `GdprRateLimiterUnitTest` (as a mock) and `GeoIpServiceUnitTest`.
8. FACT: `StepUpAuthService` and `EmailVerificationService` inject `StringRedisTemplate` with no error handling. Source: `StepUpAuthService.java:34,101-250`; `EmailVerificationService.java:51,394,405`.
9. FACT: `evict` calls sit in `@Transactional` flows with no guard. Source: `UserService.java:322,375,761,879`; `FirebaseAuthProxyService.java:1169,1257-1265`.
10. FACT: `StorageRateLimitService.hasStorageSpace` and `getGracePeriodMultiplier` call Redis outside try and are called from `checkUploadAllowed`, which `SignedUrlService:164` uses. Source: `StorageRateLimitService.java:213,238,564,738`.
11. FACT: there's no `CacheErrorHandler`/`CachingConfigurer`; `@Cacheable` sits on `CityRepository.findByName`, which `CityConverter` uses. INFERENCE: Spring's default `SimpleCacheErrorHandler` rethrows. Source: grep; `city/CityRepository.java:14`; `city/CityConverter.java:22`.
12. FACT: `RedisGeoLocationCache.get`/`put` catch errors; `evict`, `tryLock` and `releaseLock` don't. Source: `RedisGeoLocationCache.java:69-85,102-109,125,229-238`.
13. FACT: the impossible-travel check has a 500 ms timeout and fails closed after 5 failures for ADMIN, PENDING_ADMIN and COMPANY. HYPOTHESIS: an outage makes Redis GETs slow enough to hit that. Source: `SessionSecurityService.java:277-315`; the geo executor is a fixed pool of 3 (`RedisGeoLocationCache.java:43`).
14. FACT: sessions are cookie JWT plus HMAC with no Spring Session dependency, and `SocialAuthSessionService` is a `ConcurrentHashMap`. Source: `SessionSecurityService.java`; `pom.xml` grep (no spring-session); `SocialAuthSessionService.java:21`.
15. FACT: prod enables `management.health.redis`, and `UploadSystemHealthIndicator` goes DOWN when Redis fails. The Docker healthcheck hits `/api/actuator/health`. Source: `application-prod.yml:341-343`; `UploadSystemHealthIndicator.java:93-115`; `docker-compose-prod.yml:299-304`.
16. INFERENCE: the aggregate health endpoint returns 503 when DOWN (Boot default) and Docker doesn't restart unhealthy containers; no autoheal exists in the repo. Source: grep of deployment/ansible/.github for "unhealthy|autoheal" found nothing.
17. FACT: `isRedisRequired()` reads `System.getProperty("spring.profiles.active")`; prod sets the env var `SPRING_PROFILES_ACTIVE`, and `JAVA_OPTS`/entrypoint pass no `-Dspring.profiles.active`. Source: `RedisStartupConnectivityTest.java:151-156`; `docker-compose-prod.yml:269,278`; `spring-boot-entrypoint.sh:211`.
18. FACT: `RedisValidationService` catches all errors and logs "FALLING BACK TO IN-MEMORY" without doing anything. Source: `RedisValidationService.java:187-208`.
19. INFERENCE: `no-redis` fails to boot because nothing provides `StringRedisTemplate` for `StepUpAuthService`/`EmailVerificationService`. Source: `application-no-redis.yml:45,58-63`; `RedisConfiguration.java:50`.
20. FACT: `GeoLocationGdprService` keys on `geoip.cache.type=redis`, which no yml sets. Source: `GeoLocationGdprService.java:30`; grep of resources.
21. FACT: prod compose defines a single `app` service, so local fallbacks are consistent today. Source: `docker-compose-prod.yml:245`.
22. INFERENCE: Redis-backed features recover without a restart. Source: `autoReconnect(true)` in `RedisConfiguration.java:201`.
23. HYPOTHESIS: login failures during the outage surface as HTTP 500. Source: unverified exception-handler mapping.

## Corrections after verification

**Checked and confirmed**
- **#19, `no-redis` can't boot:** confirmed as far as reading code can take it. There's no `StringRedisTemplate` or `RedisConnectionFactory` bean anywhere in `src/main/java`, and `no-redis` turns off both `RedisConfiguration` and `RedisAutoConfiguration`. So the required `StringRedisTemplate` for `StepUpAuthService` and `EmailVerificationService` can't be satisfied. It stays INFERENCE until something actually starts the app; that's the job of the new `NoRedisProfileContextIT` in Plan step 1.
- **#17, fail-fast is dead in prod:** confirmed for Ansible deploys too. `ansible/roles/11-docker-compose/templates/docker-compose-prod.yml.j2:201,210` sets the profile only as an env var, and its default `JAVA_OPTS` has no `-Dspring.profiles.active`. The same template's healthcheck (line 228) also hits the aggregate `/api/actuator/health`, so Plan step 9 covers that template, as already listed.
- **#16, health returns 503:** partly confirmed. No yml sets a custom `status.http-mapping` or `order`, so Boot's default (DOWN → 503) applies. Nothing in the repo restarts unhealthy containers, but that "Docker won't restart them" part is still Docker behaviour I haven't run.

**Corrections**
1. **#23 was wrong.** Login during the outage returns **401 `error.auth.token_exchange_failed`**, not 500. `TokenExchangeService` wraps the token exchange in `catch (Exception)` (line 399), rethrows as `AuthenticationTranslatableException`, and `GlobalDefaultExceptionHandler:97` maps that to 401. The impact matrix row "Login / token exchange / refresh → FAIL" still holds, but users see an auth error, not a server error. That makes the Plan step 3 rule ("infrastructure errors never produce 401") broader: `TokenExchangeService.java` must also rethrow `DataAccessException` as a 503 before its generic catch. **Add `auth/service/TokenExchangeService.java` to Plan step 3**, with a test in the new `JwtAuthenticationFilterRedisOutageUnitTest` or a sibling `TokenExchangeServiceRedisOutageUnitTest`.
2. **#11 is unchanged** and stays INFERENCE: it relies on Spring's default `SimpleCacheErrorHandler`, not on code in this repo. Plan step 8 is still right either way.
3. **Still unverified:** #5, #13 and #22 (timing and recovery details of the Redis client, and whether slow Redis GETs trigger the travel-check lockout). They don't change the plan. The shared breaker in step 2 removes the timeout exposure whichever way they turn out, and `RedisOutageIT` (step 12) is where they get measured.
