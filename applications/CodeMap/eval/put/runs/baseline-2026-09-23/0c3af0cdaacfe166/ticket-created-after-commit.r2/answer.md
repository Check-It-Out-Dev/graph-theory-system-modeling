All compiles cleanly with no errors.

## Summary

- `src/main/java/.../support/ticket/event/SupportTicketCreatedEvent.java` (new): carries ticketId, contactEmail, ticketReference, subject, language, statusToken.
- `src/main/java/.../support/ticket/event/SupportTicketCreatedEventListener.java` (new): `onTicketCreated` sends the confirmation via `EmailService`, `@TransactionalEventListener(phase = AFTER_COMMIT)`, catches/logs failures.
- `SupportTicketService.java`: `createTicket` now publishes `SupportTicketCreatedEvent` via injected `ApplicationEventPublisher` instead of calling `EmailService` directly.
- `SupportTicketServiceUnitTest.java`: added `ApplicationEventPublisher` mock and a `createTicket` test verifying the event is published and `emailService` is never touched.
- `SupportTicketCreatedEventListenerUnitTest.java` (new): verifies the listener forwards event data to `EmailService` and swallows send failures.
- `SupportTicketService_Create_IntegrationTest.java`: updated the now-incorrect "sends confirmation email" assertion to verify no synchronous email call happens inside the still-open transaction (matches the codebase's established note that AFTER_COMMIT listeners aren't testable under rollback-based integration tests).

Tests run: `./mvnw test -Ptest -Dtest=SupportTicketServiceUnitTest,SupportTicketCreatedEventListenerUnitTest` — 56 tests, all passed, BUILD SUCCESS. Also verified `compile` and `test-compile` succeed for the whole project.