All 58 tests pass, BUILD SUCCESS.

Changes:
- `repositories/SupportTicketRepository.java`: added `findByStatusAndResolvedTimeBefore(TicketStatus, LocalDateTime)`.
- `services/SupportTicketService.java`: added `closeStaleResolvedTickets(int olderThanDays)`, transactional, closes every RESOLVED ticket older than the cutoff and returns the count.
- `services/StaleTicketCloseCronJob.java` (new): `@Scheduled`+`@SchedulerLock("support:staleTicketClose")` job, config-driven `enabled`/`olderThanDays`/cron via `support.ticket.stale-close.*` properties, catches and logs failures.
- `application.yml`: added `support.ticket.stale-close` defaults (cron `0 30 3 * * *`, enabled `true`, days `14`).
- Tests: extended `SupportTicketServiceUnitTest` with `CloseStaleResolvedTicketsTests`, added new `StaleTicketCloseCronJobUnitTest`.

Ran `./mvnw test -Ptest -Dtest=StaleTicketCloseCronJobUnitTest,SupportTicketServiceUnitTest` — 58 tests, 0 failures, BUILD SUCCESS.