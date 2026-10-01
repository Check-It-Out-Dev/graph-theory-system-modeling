Compiles cleanly.

## Summary

- `src/main/java/com/sm/instagram/platform/support/ticket/repositories/SupportTicketRepository.java`: added `findByStatusAndResolvedTimeBefore(TicketStatus, LocalDateTime)`.
- `src/main/java/com/sm/instagram/platform/support/ticket/services/SupportTicketService.java`: added `@Transactional int closeStaleResolvedTickets(int olderThanDays)`, moving RESOLVED tickets past the cutoff to CLOSED and saving them.
- `src/main/java/com/sm/instagram/platform/support/ticket/services/StaleTicketCloseCronJob.java` (new): nightly `@Scheduled`/`@SchedulerLock` job (`support:staleTicketClose`), driven by `support.ticket.stale-close.{enabled,days,cron}` (defaults `true`/`14`/`0 30 3 * * *`), logging failures instead of rethrowing — follows `NoConsentAccountCleanupCronJob`/`DeferredDeletionCronJob` pattern.
- `src/test/java/com/sm/instagram/platform/unit/service/SupportTicketServiceUnitTest.java`: added `CloseStaleResolvedTickets` nested tests.
- `src/test/java/com/sm/instagram/platform/unit/service/StaleTicketCloseCronJobUnitTest.java` (new): enabled/disabled/exception-swallowing tests.

Ran `./mvnw -q test -Ptest -Dtest=SupportTicketServiceUnitTest,StaleTicketCloseCronJobUnitTest -Dsurefire.failIfNoSpecifiedTests=false` — all pass (new nested suites: 2/2 and 3/3, 0 failures). Also ran `./mvnw -q compile` — builds clean.