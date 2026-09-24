All rule checks are satisfied: `@SchedulerLock(name="support:staleTicketClose")`, property-driven cron with defaults, `enabled` flag, class named `*CronJob`, delegates to the service, catches/logs failures; constructor injection via `@RequiredArgsConstructor`; no schema/DTO/controller changes needed since this is an internal batch job.

## Summary
- `SupportTicketRepository.java`: added `findByStatusAndResolvedTimeBefore(TicketStatus, LocalDateTime)`.
- `SupportTicketService.java`: added `closeStaleResolvedTickets(int olderThanDays)` — finds RESOLVED tickets past the cutoff, sets them CLOSED, saves, returns the count.
- `StaleTicketCloseCronJob.java` (new): nightly job with `enabled`/`olderThanDays`/`cron` from `support.ticket.stale-close.*` properties, `@SchedulerLock(name="support:staleTicketClose")`, catches and logs service failures.
- `application.yml`: added `support.ticket.stale-close` block (cron `0 30 3 * * *`, enabled `true`, days `14`).
- Tests: extended `SupportTicketServiceUnitTest` with `closeStaleResolvedTickets` cases (main path, cutoff value, empty path); added `StaleTicketCloseCronJobUnitTest` (enabled/disabled/exception-swallowed paths).
- Ran `SupportTicketServiceUnitTest` and `StaleTicketCloseCronJobUnitTest` — passed. Ran `test-compile -Ptest` — passed.