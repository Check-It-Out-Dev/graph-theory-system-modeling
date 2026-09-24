All changes are scoped to the support ticket feature. Working tree is left uncommitted as requested.

## Summary

- `support/ticket/event/SupportTicketResolvedEvent.java` — new `ApplicationEvent` carrying ticketId, contactEmail, ticketReference.
- `support/ticket/event/SupportTicketResolvedEventListener.java` — `@TransactionalEventListener(AFTER_COMMIT)`, delegates to `TicketNotificationPort`, catches/logs failures.
- `support/ticket/port/TicketNotificationPort.java` — replaceable notification port (`notifyResolved`).
- `support/ticket/adapter/EmailTicketNotificationAdapter.java` — e-mail implementation via `EmailService#sendEmail`, subject includes ticket reference.
- `support/ticket/services/SupportTicketService.java` — constructor now takes `ApplicationEventPublisher`; `updateTicketStatus` publishes `SupportTicketResolvedEvent` only when new status is RESOLVED.
- `unit/service/SupportTicketServiceUnitTest.java` — added `eventPublisher` mock, tests asserting the event is published (with correct fields, captured via `ArgumentCaptor`) on RESOLVED and not published on other transitions.
- New tests: `SupportTicketResolvedEventListenerUnitTest` (delegates to port; swallows port failure) and `EmailTicketNotificationAdapterUnitTest` (subject/text contain ticket reference).

Tests run: `SupportTicketServiceUnitTest`, `SupportTicketResolvedEventListenerUnitTest`, `EmailTicketNotificationAdapterUnitTest`, `SupportTicketControllerFullUnitTest` — all passed; `test-compile -Ptest` green.