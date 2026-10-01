## Summary

- Added `SupportTicketCreatedEvent` (`src/main/java/.../support/ticket/event/`) carrying the six ticket fields.
- Added `SupportTicketCreatedEventListener`, `@TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)`, sends the confirmation via `EmailService#sendTicketCreationConfirmation`, catches/logs failures.
- `SupportTicketService`: constructor-injected `ApplicationEventPublisher`; `createTicket` now publishes `SupportTicketCreatedEvent` instead of calling `EmailService` directly.
- Updated `SupportTicketServiceUnitTest`: mocked `ApplicationEventPublisher`, added `CreateTicketTests` nested class capturing the published event and asserting no direct `EmailService` call.
- Added `SupportTicketCreatedEventListenerUnitTest` covering the happy path and the swallowed-exception path.
- Updated `SupportTicketService_Create_IntegrationTest` (its transaction rolls back, so `AFTER_COMMIT` never fires there): replaced the email-mock assertion with `@RecordApplicationEvents`/`ApplicationEvents` verifying the event is published with the right ticket data.

Tests run: `mvnw test -Ptest -Dtest=SupportTicketServiceUnitTest,SupportTicketCreatedEventListenerUnitTest` (58 tests, pass) and `-Dtest=SupportTicketControllerFullUnitTest,SupportTicketEntityUnitTest,SupportTicketModelsUnitTest` (337 tests, pass); `mvnw test-compile -Ptest` is green.