Compiles cleanly, and no other callers of the `SupportTicketService` constructor exist to update.

## Summary

- `src/main/java/com/sm/instagram/platform/support/ticket/event/SupportTicketCreatedEvent.java` — new event carrying ticketId, contactEmail, ticketReference, subject, language, statusToken.
- `src/main/java/com/sm/instagram/platform/support/ticket/event/SupportTicketCreatedEventListener.java` — new `@TransactionalEventListener(phase = AFTER_COMMIT)` that sends the confirmation via `EmailService`, catching/logging its own failures.
- `src/main/java/com/sm/instagram/platform/support/ticket/services/SupportTicketService.java` — `createTicket` now publishes `SupportTicketCreatedEvent` via `ApplicationEventPublisher` instead of calling `EmailService` directly (constructor gained the publisher dependency).
- `src/test/java/com/sm/instagram/platform/support/ticket/event/SupportTicketCreatedEventListenerUnitTest.java` — new unit test covering the successful send and the swallowed-failure case.

Tests run: `SupportTicketCreatedEventListenerUnitTest` — passed. Full `mvn -Ptest compile` — clean.