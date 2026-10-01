Changes made:
- `FaqRepository.java` — added `findByActiveFalseAndLastUpdateTimeBefore(LocalDateTime cutoff)`.
- `FaqService.java` — added `purgeInactiveOlderThan(int days)` deleting stale inactive FAQs and returning the count.
- `InactiveFaqPurgeCronJob.java` (new) — weekly `@Scheduled`/`@SchedulerLock` job (`faq:inactivePurge`), property-driven `enabled`/`olderThanDays`/cron (defaults `true`/180/`0 0 4 * * SUN`), catches and logs service failures.
- `application.yml` — new `support.faq.purge` section with the same defaults.
- `FaqServiceUnitTest.java`, `InactiveFaqPurgeCronJobUnitTest.java` (new, `@ExtendWith(MockitoExtension.class)`, no `lenient`/`LENIENT`) — cover deletion count, cutoff computation, empty case, enabled/disabled/exception-swallowed paths.

Tests run: `FaqServiceUnitTest`, `InactiveFaqPurgeCronJobUnitTest`, `FaqUnitTest` — all passed; `test-compile -Ptest` green.