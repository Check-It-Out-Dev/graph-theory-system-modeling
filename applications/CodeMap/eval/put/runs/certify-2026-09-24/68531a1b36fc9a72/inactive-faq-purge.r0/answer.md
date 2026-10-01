All changes are scoped to the task. Working tree left uncommitted as instructed.

**Summary:**
- `FaqRepository.java`: added `findByActiveFalseAndLastUpdateTimeBefore(LocalDateTime cutoff)`.
- `FaqService.java`: added `purgeInactiveOlderThan(int days)`, which deletes and returns the count of inactive FAQs older than the cutoff.
- `InactiveFaqPurgeCronJob.java` (new): weekly `@Scheduled`/`@SchedulerLock("support:faqPurge")` job with `enabled` (`support.faq.purge.enabled`, default true) and `olderThanDays` (`support.faq.purge.days`, default 180); catches and logs service failures.
- `application.yml`: documented the new `support.faq.purge.*` properties (enabled/cron/days) alongside the existing consent-cleanup block.
- `FaqServiceUnitTest.java` (new) and `InactiveFaqPurgeCronJobUnitTest.java` (new): cover the main/empty/error paths and the cutoff computation, `@ExtendWith(MockitoExtension.class)`, strict Mockito (no `LENIENT`).

Tests run: `FaqServiceUnitTest`, `InactiveFaqPurgeCronJobUnitTest`, `FaqUnitTest` — all pass; `mvn test-compile -Ptest` — green.