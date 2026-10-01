All 6 new tests pass (3+3), and the existing `FaqUnitTest` (24 tests across nested classes) also passed with 0 failures/errors, confirming nothing broke.

## Summary

- `src/main/java/.../support/faq/repositories/FaqRepository.java`: added `findByActiveFalseAndLastUpdateTimeBefore(LocalDateTime cutoff)`.
- `src/main/java/.../support/faq/services/FaqService.java`: added `purgeInactiveOlderThan(int days)`, deleting and counting inactive FAQs older than the cutoff.
- `src/main/java/.../support/faq/services/InactiveFaqPurgeCronJob.java` (new): weekly `@Scheduled`/`@SchedulerLock` job (`support:faqPurge`), configurable via `support.faq.purge.{enabled,days,cron}`, disabled skip and try/catch-log on failure, following `AnonymousConsentCleanupCronJob`'s pattern.
- `src/main/resources/application.yml`: added `support.faq.purge` block (`enabled: true`, `cron: "0 0 4 * * SUN"`, `days: 180`).
- New tests: `FaqServiceUnitTest` (3 tests) and `InactiveFaqPurgeCronJobUnitTest` (3 tests), mirroring existing repo conventions (Mockito, `ReflectionTestUtils` for `@Value` fields).
- Ran `FaqServiceUnitTest`, `InactiveFaqPurgeCronJobUnitTest`, `FaqUnitTest` via `-Ptest`: all pass, 0 failures/errors.