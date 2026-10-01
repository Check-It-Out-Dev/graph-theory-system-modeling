Compiles cleanly with no errors.

## Summary

- `src/main/java/.../support/ticket/repositories/SupportTicketRepository.java`: added `findByStatusAndResolvedTimeBefore(TicketStatus, LocalDateTime)`.
- `src/main/java/.../support/ticket/services/SupportTicketService.java`: added `closeStaleResolvedTickets(int olderThanDays)`, closing every RESOLVED ticket whose `resolvedTime` is older than the cutoff and returning the count.
- `src/main/java/.../support/ticket/services/StaleTicketCloseCronJob.java` (new): nightly `@Scheduled`/`@SchedulerLock`-guarded job (`support:staleTicketClose`), property-driven `enabled`/`olderThanDays`/cron (defaults `true`/14/`0 30 3 * * *`), catches and logs service failures.
- `src/test/java/.../unit/service/StaleTicketCloseCronJobUnitTest.java` (new): covers enabled/disabled/exception-swallowing/configured-days behavior.
- `src/test/java/.../unit/service/SupportTicketServiceStaleCloseUnitTest.java` (new): covers closing stale tickets, cutoff computation, and the zero-tickets case.

Tests run: `StaleTicketCloseCronJobUnitTest`, `SupportTicketServiceStaleCloseUnitTest`, `SupportTicketServiceUnitTest`, `SupportTicketControllerFullUnitTest` — all pass. Full `compile`/`test-compile` (`-Ptest`) also succeeds.