All checks pass. Summary of changes:

- `src/main/java/.../support/ticket/repositories/SupportTicketRepository.java` — added `findByStatusAndResolvedTimeBefore(TicketStatus, LocalDateTime)`.
- `src/main/java/.../support/ticket/services/SupportTicketService.java` — added `closeStaleResolvedTickets(int olderThanDays)`, transactional, closes and saves each stale RESOLVED ticket, returns count.
- `src/main/java/.../support/ticket/services/StaleTicketCloseCronJob.java` (new) — nightly `@Scheduled`/`@SchedulerLock("support:staleTicketClose")` job, property-driven `enabled`/`olderThanDays`/cron, delegates to service, catches and logs failures.
- `src/main/resources/application.yml` — added `support.ticket.stale-close.{enabled,days,cron}` defaults (true/14/`0 30 3 * * *`).
- New tests: `SupportTicketServiceStaleCloseUnitTest.java` (3 tests: closes+counts, cutoff computation, empty path) and `StaleTicketCloseCronJobUnitTest.java` (3 tests: enabled delegation, disabled no-op, exception swallowed).

Ran: `SupportTicketServiceStaleCloseUnitTest`, `StaleTicketCloseCronJobUnitTest`, `SupportTicketServiceUnitTest`, `SupportTicketControllerFullUnitTest` — all pass; `test-compile -Ptest` green.