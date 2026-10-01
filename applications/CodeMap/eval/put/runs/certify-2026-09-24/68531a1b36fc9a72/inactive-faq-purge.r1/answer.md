Summary:
- `FaqRepository.java`: added `findByActiveFalseAndLastUpdateTimeBefore(LocalDateTime cutoff)`.
- `FaqService.java`: added `purgeInactiveOlderThan(int days)` — finds and deletes stale inactive FAQs, returns count.
- `InactiveFaqPurgeCronJob.java` (new): weekly `@Scheduled`/`@SchedulerLock` job, property-driven `enabled`/`olderThanDays`/cron, catches and logs failures.
- `application.yml`: added `support.faq.purge` defaults (enabled, cron, days).
- New tests: `FaqServiceUnitTest.java`, `InactiveFaqPurgeCronJobUnitTest.java` — cover delete/return-count, empty result, enabled/disabled, and failure-swallowing paths.
- Ran: `FaqServiceUnitTest`, `InactiveFaqPurgeCronJobUnitTest`, `FaqUnitTest` (70 tests, all pass); `test-compile -Ptest` green.