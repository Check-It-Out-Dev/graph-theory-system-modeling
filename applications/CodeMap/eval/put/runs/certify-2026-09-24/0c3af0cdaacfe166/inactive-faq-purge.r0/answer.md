All 6 new tests pass and existing FaqUnitTest suite is unaffected (0 failures/errors).

## Summary

- `src/main/java/com/sm/instagram/platform/support/faq/repositories/FaqRepository.java` — added `findByActiveFalseAndLastUpdateTimeBefore(LocalDateTime cutoff)`.
- `src/main/java/com/sm/instagram/platform/support/faq/services/FaqService.java` — added `purgeInactiveOlderThan(int days)`: finds and deletes inactive FAQs older than the cutoff, returns the count.
- `src/main/java/com/sm/instagram/platform/support/faq/services/InactiveFaqPurgeCronJob.java` (new) — weekly `@Scheduled`/`@SchedulerLock` job `purgeInactiveFaqs()`, config via `support.faq.purge.enabled` (default true), `support.faq.purge.days` (default 180), `support.faq.purge.cron` (default `0 0 4 * * SUN`); catches and logs failures, skips when disabled — modeled on `AnonymousConsentCleanupCronJob`.
- `src/test/java/com/sm/instagram/platform/unit/service/FaqServiceUnitTest.java` (new) — 3 tests for `purgeInactiveOlderThan`.
- `src/test/java/com/sm/instagram/platform/unit/service/InactiveFaqPurgeCronJobUnitTest.java` (new) — 3 tests covering enabled/disabled/exception-swallowed cases.

Tests run: `./mvnw -q test -Ptest -Dtest=FaqServiceUnitTest,InactiveFaqPurgeCronJobUnitTest,FaqUnitTest -Dsurefire.failIfNoSpecifiedTests=false` — all pass (6 new tests, 0 failures; existing `FaqUnitTest` suite unaffected).