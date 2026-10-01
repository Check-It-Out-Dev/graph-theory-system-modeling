No manual instantiations exist — Spring autowires the new `ApplicationEventPublisher` param automatically.

## Summary

- `event/SupportTicketResolvedEvent.java` (new): `ApplicationEvent` carrying ticketId, contactEmail, ticketReference.
- `port/TicketNotificationPort.java` (new): `notifyResolved(contactEmail, ticketReference)`.
- `adapter/EmailTicketNotificationAdapter.java` (new): implements the port via `EmailService#sendEmail`, subject includes the reference.
- `event/SupportTicketResolvedEventListener.java` (new): `@TransactionalEventListener(AFTER_COMMIT)`, constructor-injects the port, catches and logs failures.
- `services/SupportTicketService.java`: constructor now takes `ApplicationEventPublisher`; `updateTicketStatus` publishes `SupportTicketResolvedEvent` only when the new status is `RESOLVED`.
- Tests: updated `SupportTicketServiceUnitTest` (new mock + 2 new cases for publish/no-publish on RESOLVED vs. other transitions), added `SupportTicketResolvedEventListenerUnitTest` and `EmailTicketNotificationAdapterUnitTest`.
- Ran `SupportTicketServiceUnitTest`, `SupportTicketResolvedEventListenerUnitTest`, `EmailTicketNotificationAdapterUnitTest` via `-Ptest` — all passed (the one ERROR log line is the expected, caught failure in the "port fails" test).