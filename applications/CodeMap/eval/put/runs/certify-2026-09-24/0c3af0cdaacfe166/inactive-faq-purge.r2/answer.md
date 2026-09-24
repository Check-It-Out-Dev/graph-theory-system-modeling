All existing FAQ tests pass (no failures reported). Change is complete.

## Summary

- `src/main/java/com/sm/instagram/platform/support/faq/repositories/FaqRepository.java`: added `findByActiveFalseAndLastUpdateTimeBefore(LocalDateTime cutoff)`.
- `src/main/java/com/sm/instagram/platform/support/faq/services/FaqService.java`: added `purgeInactiveOlderThan(int days)`, deletes and returns the count of inactive FAQs older than the cutoff.
- `src/main/java/com/sm/instagram/platform/support/faq/services/InactiveFaqPurgeCronJob.java` (new): weekly `@Scheduled`/`@SchedulerLock` job (`support.faq.purge.cron`, default `0 0 4 * * SUN`), with `enabled` (`support.faq.purge.enabled`, default true) and `olderThanDays` (`support.faq.purge.days`, default 180); catches and logs failures.
- Tests added: `src/test/java/.../unit/service/FaqPurgeUnitTest.java` (FaqService.purgeInactiveOlderThan) and `InactiveFaqPurgeCronJobUnitTest.java` (job enable/disable/error-swallow). Ran `FaqPurgeUnitTest`, `InactiveFaqPurgeCronJobUnitTest`, and the pre-existing `FaqUnitTest`, `AnonymousConsentCleanupCronJobUnitTest` — all pass (no failures/errors in surefire reports).