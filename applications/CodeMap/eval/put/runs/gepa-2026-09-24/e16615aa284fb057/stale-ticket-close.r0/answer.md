All checks pass: `@SchedulerLock` present with `area:jobName` format, no LENIENT/lenient usage, and test packages match their directories.

## Summary

- `src/main/java/.../support/ticket/repositories/SupportTicketRepository.java` — added `findByStatusAndResolvedTimeBefore(TicketStatus, LocalDateTime)`.
- `src/main/java/.../support/ticket/services/SupportTicketService.java` — added `closeStaleResolvedTickets(int olderThanDays)`, closing and saving every RESOLVED ticket older than the cutoff, returning the count.
- `src/main/java/.../support/ticket/services/StaleTicketCloseCronJob.java` (new) — nightly `@Scheduled`/`@SchedulerLock("support:ticketStaleClose")` job, property-driven cron/enabled/days, catches and logs service failures.
- `src/main/resources/application.yml` — added `support.ticket.stale-close` defaults (`enabled: true`, `days: 14`, `cron: "0 30 3 * * *"`).
- New tests: `SupportTicketServiceCloseStaleResolvedTicketsUnitTest.java`, `StaleTicketCloseCronJobUnitTest.java` (feature-package, strict Mockito).

Tests run: new tests (6 total) all pass; existing `SupportTicketServiceUnitTest` unaffected (all green); `test-compile -Ptest` clean.