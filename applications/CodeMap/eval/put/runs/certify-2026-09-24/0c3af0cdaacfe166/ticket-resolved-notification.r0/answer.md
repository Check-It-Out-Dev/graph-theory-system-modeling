All 6 tests in `StatusTransitionTests` (including the 2 new event-publishing tests) passed, confirming the change works.

## Summary

- `src/main/java/.../support/ticket/event/SupportTicketResolvedEvent.java` (new): event with `ticketId`, `contactEmail`, `ticketReference`.
- `src/main/java/.../support/ticket/port/TicketNotificationPort.java` (new): `notifyResolved(contactEmail, ticketReference)` port.
- `src/main/java/.../support/ticket/adapter/EmailTicketNotificationAdapter.java` (new): implements the port via `EmailService#sendEmail`, subject includes the ticket reference.
- `src/main/java/.../support/ticket/event/SupportTicketResolvedEventListener.java` (new): `@TransactionalEventListener(AFTER_COMMIT)`, catches and logs port failures.
- `src/main/java/.../support/ticket/services/SupportTicketService.java`: constructor now takes `ApplicationEventPublisher`; `updateTicketStatus` publishes `SupportTicketResolvedEvent` only when the new status is RESOLVED.
- `src/test/java/.../unit/service/SupportTicketServiceUnitTest.java`: added `eventPublisher` mock plus two tests (publishes on RESOLVED, doesn't publish otherwise).
- `src/test/java/.../support/ticket/event/SupportTicketResolvedEventListenerUnitTest.java` (new), `src/test/java/.../support/ticket/adapter/EmailTicketNotificationAdapterUnitTest.java` (new).

Ran `./mvnw -q test -Ptest -Dtest=SupportTicketServiceUnitTest,SupportTicketResolvedEventListenerUnitTest,EmailTicketNotificationAdapterUnitTest` — all passed (SupportTicketServiceUnitTest's nested classes incl. StatusTransitionTests: 6/6, listener test: 2/2, adapter test: 1/1).