No `@PreAuthorize`/DTO/Liquibase/i18n/ports rules apply here (no endpoints, entities, or schema changed). This is a pure repository + service + scheduled job addition, matching scope.

## Summary

- `FaqRepository.java`: added `findByActiveFalseAndLastUpdateTimeBefore(LocalDateTime cutoff)`.
- `FaqService.java`: added `purgeInactiveOlderThan(int days)`, deleting and counting inactive FAQs older than the cutoff.
- `InactiveFaqPurgeCronJob.java` (new): weekly `@Scheduled` job with `@SchedulerLock(name = "faq:inactivePurge")`, properties `support.faq.purge.enabled` (default `true`), `support.faq.purge.days` (default `180`), `support.faq.purge.cron` (default `0 0 4 * * SUN`); logs and swallows service failures.
- `InactiveFaqPurgeCronJobUnitTest.java` (new): covers enabled path (calls service with configured days), disabled path (no interaction), and exception swallowing.
- `FaqService_PurgeUnitTest.java` (new): asserts the cutoff passed to the repository (via `ArgumentCaptor`), the delete count returned, and the empty-result path.

Tests run: `InactiveFaqPurgeCronJobUnitTest`, `FaqService_PurgeUnitTest` — 5/5 passed. `mvnw test-compile -Ptest` — green.