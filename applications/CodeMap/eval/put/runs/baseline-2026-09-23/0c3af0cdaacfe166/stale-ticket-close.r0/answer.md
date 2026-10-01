All tests pass, 0 failures/errors across all nested classes. The implementation is complete.

## Summary

- `src/main/java/com/sm/instagram/platform/support/ticket/repositories/SupportTicketRepository.java`: added `findByStatusAndResolvedTimeBefore(TicketStatus, LocalDateTime)`.
- `src/main/java/com/sm/instagram/platform/support/ticket/services/SupportTicketService.java`: added `closeStaleResolvedTickets(int olderThanDays)`, transitioning every stale RESOLVED ticket to CLOSED and saving it.
- `src/main/java/com/sm/instagram/platform/support/ticket/services/StaleTicketCloseCronJob.java` (new): nightly `@Scheduled`/`@SchedulerLock` job (`support:staleTicketClose`), configurable via `support.ticket.stale-close.{enabled,days,cron}` (defaults `true`/`14`/`0 30 3 * * *`), skips when disabled, logs failures instead of rethrowing — modeled on `NoConsentAccountCleanupCronJob`.
- Tests: added `CloseStaleResolvedTicketsTests` nested class to `SupportTicketServiceUnitTest.java`, and new `StaleTicketCloseCronJobUnitTest.java`.
- Ran `./mvnw -q test -Ptest -Dtest=SupportTicketServiceUnitTest,StaleTicketCloseCronJobUnitTest` — all tests pass (0 failures/errors).