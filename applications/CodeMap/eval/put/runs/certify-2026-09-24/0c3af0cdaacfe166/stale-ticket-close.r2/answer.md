Compiles cleanly and all tests pass.

## Summary

- `src/main/java/com/sm/instagram/platform/support/ticket/repositories/SupportTicketRepository.java`: added `findByStatusAndResolvedTimeBefore(TicketStatus status, LocalDateTime cutoff)`.
- `src/main/java/com/sm/instagram/platform/support/ticket/services/SupportTicketService.java`: added `@Transactional int closeStaleResolvedTickets(int olderThanDays)` — finds RESOLVED tickets whose `resolvedTime` is before now-minus-days, sets each to CLOSED, saves, and returns the count.
- `src/main/java/com/sm/instagram/platform/support/ticket/services/StaleTicketCloseCronJob.java` (new): nightly `@Scheduled`/`@SchedulerLock` job (`support:staleTicketClose`), configurable via `support.ticket.stale-close.{enabled,days,cron}` (defaults `true`, `14`, `0 30 3 * * *`); logs and swallows failures.
- Tests added: `CloseStaleResolvedTicketsTests` nested class in `SupportTicketServiceUnitTest.java` (2 tests), and new `StaleTicketCloseCronJobUnitTest.java` (3 tests).
- Ran `./mvnw -q test -Ptest -Dtest=SupportTicketServiceUnitTest,StaleTicketCloseCronJobUnitTest` — all 55 tests pass. Also ran `./mvnw -q compile` — clean build.