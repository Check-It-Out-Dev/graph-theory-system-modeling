Summary:
- `SupportTicketRepository.java`: added `findByStatusAndResolvedTimeBefore(TicketStatus, LocalDateTime)`.
- `SupportTicketService.java`: added `closeStaleResolvedTickets(int olderThanDays)`, closing every RESOLVED ticket past cutoff and returning the count.
- `StaleTicketCloseCronJob.java` (new): nightly `@Scheduled`/`@SchedulerLock("support:staleTicketClose")` job, property-driven `enabled`/`olderThanDays`/cron (defaults true/14/`0 30 3 * * *`), catches and logs service failures.
- `application.yml`: added `support.ticket.stale-close.{enabled,cron,days}` defaults.
- `SupportTicketServiceUnitTest.java`: added tests for `closeStaleResolvedTickets` (closes+counts stale tickets with captured cutoff, and empty path saves nothing).
- `StaleTicketCloseCronJobUnitTest.java` (new): tests enabled/disabled/failure-swallowed paths.

Tests run: `SupportTicketServiceUnitTest`, `StaleTicketCloseCronJobUnitTest` — passed; `mvnw test-compile -Ptest` — green.