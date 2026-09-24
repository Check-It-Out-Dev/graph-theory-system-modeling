Changes complete.

- `src/main/java/com/sm/instagram/platform/support/ticket/event/SupportTicketCreatedEvent.java` (new): event carrying ticket id, contact email, reference, subject, language, status token.
- `src/main/java/com/sm/instagram/platform/support/ticket/event/SupportTicketCreatedEventListener.java` (new): `@TransactionalEventListener(phase = AFTER_COMMIT)` that sends the confirmation via `EmailService`, catching/logging failures.
- `src/main/java/com/sm/instagram/platform/support/ticket/services/SupportTicketService.java`: `createTicket` now publishes `SupportTicketCreatedEvent` via injected `ApplicationEventPublisher` instead of calling `EmailService` inline.
- `src/test/java/.../support/ticket/event/SupportTicketCreatedEventListenerUnitTest.java` (new): verifies email is sent with event data and failures are swallowed.
- `src/test/java/.../unit/service/SupportTicketServiceUnitTest.java`: added `eventPublisher` mock and a `createTicket` test verifying the event is published and `EmailService` is never touched directly.
- `src/test/java/.../integration/service/supportticket/SupportTicketService_Create_IntegrationTest.java`: updated the now-invalid "sends email inline" assertion to verify event publication via `@RecordApplicationEvents`/`ApplicationEvents` (the old assertion could no longer pass since the send now only happens AFTER_COMMIT, and the test transaction rolls back).

Tests run: `SupportTicketServiceUnitTest` (all nested classes, including new `CreateTicketTests`) and `SupportTicketCreatedEventListenerUnitTest` — all green, 0 failures/errors.