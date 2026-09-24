Confirmed — `@SchedulerLock(name = "support:staleTicketClose", ...)` is present, formatted like the `NoConsentAccountCleanupCronJob` exemplar. All checks pass.

## Summary

- `SupportTicketRepository.java`: added `findByStatusAndResolvedTimeBefore(TicketStatus, LocalDateTime)`.
- `SupportTicketService.java`: added `closeStaleResolvedTickets(int olderThanDays)`, closing every RESOLVED ticket resolved before `now - olderThanDays` and returning the count.
- `StaleTicketCloseCronJob.java` (new, `support.ticket.services`): nightly `@Scheduled`/`@SchedulerLock("support:staleTicketClose")` job with `enabled` (`support.ticket.stale-close.enabled`, default true) and `olderThanDays` (`support.ticket.stale-close.days`, default 14) fields, cron from `support.ticket.stale-close.cron` (default `0 30 3 * * *`); catches and logs service failures.
- `application.yml`: added the `support.ticket.stale-close` property block.
- New tests: `SupportTicketServiceStaleTicketCloseUnitTest.java` and `StaleTicketCloseCronJobUnitTest.java` (both `@ExtendWith(MockitoExtension.class)`, strict, same package as the classes under test).

Tests run: `SupportTicketServiceStaleTicketCloseUnitTest`, `StaleTicketCloseCronJobUnitTest`, `SupportTicketServiceUnitTest` — 59/59 passed. `test-compile -Ptest` — green.