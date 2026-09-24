All 58 tests passed, BUILD SUCCESS. That includes both my new test classes.

## Summary

- `SupportTicketRepository.java`: added `findByStatusAndResolvedTimeBefore(TicketStatus, LocalDateTime)`.
- `SupportTicketService.java`: added `closeStaleResolvedTickets(int olderThanDays)`, closing every RESOLVED ticket whose `resolvedTime` is older than the cutoff and returning the count.
- `StaleTicketCloseCronJob.java` (new, in `support/ticket/services`): nightly `@Scheduled`/`@SchedulerLock` job (`support:staleTicketClose`), reads `support.ticket.stale-close.{enabled,days,cron}` (defaults `true`/`14`/`0 30 3 * * *`), calls the service and logs failures without rethrowing.
- Tests: added `closeStaleResolvedTickets` cases to `SupportTicketServiceUnitTest.java`, and new `StaleTicketCloseCronJobUnitTest.java` following `AnonymousConsentCleanupCronJobUnitTest` pattern.
- Ran `./mvnw test -Ptest -Dtest=SupportTicketServiceUnitTest,StaleTicketCloseCronJobUnitTest`: 58 tests, 0 failures, BUILD SUCCESS.