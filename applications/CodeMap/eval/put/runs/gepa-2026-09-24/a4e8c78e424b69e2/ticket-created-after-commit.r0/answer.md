All tests pass (170/170, BUILD SUCCESS). No new i18n keys, DTOs, or schema changes were needed since this is a plumbing fix.

## Summary

- **`src/main/java/.../support/ticket/event/SupportTicketCreatedEvent.java`** (new): event carrying `ticketId`, `contactEmail`, `ticketReference`, `subject`, `language`, `statusToken`.
- **`src/main/java/.../support/ticket/event/SupportTicketCreatedEventListener.java`** (new): `@TransactionalEventListener(phase = AFTER_COMMIT)` that calls `EmailService#sendTicketCreationConfirmation`, catching/logging any failure.
- **`SupportTicketService.java`**: added `ApplicationEventPublisher` to the constructor; `createTicket` now publishes `SupportTicketCreatedEvent` after saving instead of calling `EmailService` directly.
- **`SupportTicketServiceUnitTest.java`**: added the `ApplicationEventPublisher` mock (constructor dependent) and a new `createTicket` test asserting the event is published with the right data and `EmailService` is never called directly.
- **`SupportTicketCreatedEventListenerUnitTest.java`** (new): verifies the listener sends the confirmation with the event's data and swallows send failures.

Tests run: `SupportTicketServiceUnitTest`, `SupportTicketCreatedEventListenerUnitTest`, `SupportTicketControllerFullUnitTest`, `TicketAccessTokenServiceUnitTest` — 170/170 passed, BUILD SUCCESS.