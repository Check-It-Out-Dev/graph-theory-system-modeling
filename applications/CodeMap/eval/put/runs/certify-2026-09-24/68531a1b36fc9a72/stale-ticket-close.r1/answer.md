All expected files, nothing unrelated. No schema, DTO, or endpoint changes needed since this task doesn't touch entities structurally, controllers, or exceptions.

## Summary

- `SupportTicketRepository.java`: added `findByStatusAndResolvedTimeBefore(TicketStatus, LocalDateTime)`.
- `SupportTicketService.java`: added `closeStaleResolvedTickets(int olderThanDays)` — closes every RESOLVED ticket past the cutoff and returns the count.
- `StaleTicketCloseCronJob.java` (new): nightly `@Scheduled`/`@SchedulerLock` job (`support:staleTicketClose`), property-driven cron/enabled/days, delegates to the service and logs failures without rethrowing.
- `application.yml`: added `support.ticket.stale-close.{enabled,cron,days}` defaults matching the code's own `@Value` defaults.
- New tests: `SupportTicketServiceUnitTest` (main path, cutoff computation, empty path) and `StaleTicketCloseCronJobUnitTest` (enabled/disabled/failure-swallowed), both in the feature package, `@ExtendWith(MockitoExtension.class)`, no `LENIENT`.

Tests run: new tests (6/6 pass), existing `SupportTicketServiceUnitTest`/`TicketAccessTokenServiceUnitTest` (pass), `mvnw test-compile -Ptest` (green).