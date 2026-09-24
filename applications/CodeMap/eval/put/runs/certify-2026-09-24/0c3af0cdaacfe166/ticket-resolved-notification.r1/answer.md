Maven ran quietly (`-q`) with no BUILD FAILURE output, meaning all tests passed (the stack trace shown is just the listener logging the expected caught exception, per test design). Build succeeded.

## Summary

- `src/main/java/.../support/ticket/event/SupportTicketResolvedEvent.java` — new event carrying ticket id, contact email, reference.
- `src/main/java/.../support/ticket/port/TicketNotificationPort.java` — new port with `notifyResolved(contactEmail, ticketReference)`.
- `src/main/java/.../support/ticket/adapter/EmailTicketNotificationAdapter.java` — e-mail implementation using `EmailService#sendEmail`, subject includes the ticket reference.
- `src/main/java/.../support/ticket/event/SupportTicketResolvedEventListener.java` — `@TransactionalEventListener(AFTER_COMMIT)`, catches and logs port failures.
- `src/main/java/.../support/ticket/services/SupportTicketService.java` — injects `ApplicationEventPublisher`, publishes `SupportTicketResolvedEvent` in `updateTicketStatus` only when the new status is RESOLVED.
- `src/test/java/.../unit/service/SupportTicketServiceUnitTest.java` — added `ApplicationEventPublisher` mock and two tests (publishes on RESOLVED, doesn't publish otherwise).
- `src/test/java/.../unit/service/SupportTicketResolvedEventListenerUnitTest.java` — new test verifying the port is called and that port failures don't propagate.

Ran: `./mvnw -q test -Ptest -Dtest=SupportTicketServiceUnitTest,SupportTicketResolvedEventListenerUnitTest` — build succeeded, all tests passed (the printed stack trace is the listener's own caught-and-logged exception from the failure test case, by design).