All changes are in scope for the task.

## Summary

- Added `support/ticket/event/SupportTicketCreatedEvent.java` and `SupportTicketCreatedEventListener.java` (new package), the listener sends the confirmation via `EmailService#sendTicketCreationConfirmation` on `@TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)`, catching and logging failures.
- `SupportTicketService.createTicket` now injects `ApplicationEventPublisher` (added to the existing constructor) and publishes `SupportTicketCreatedEvent` instead of calling `EmailService` directly.
- Updated `SupportTicketServiceUnitTest` (added `eventPublisher` mock, new `CreateTicketTests` nested class asserting the published event's fields and verifying no direct `EmailService` interaction).
- Added `SupportTicketCreatedEventListenerUnitTest` covering the success path and the swallowed-exception path.
- Updated the integration test `sendsConfirmationEmailOnTicketCreation` → `doesNotSendConfirmationEmailInsideTheTransaction`, since it previously asserted the now-removed synchronous send.

Tests run: `SupportTicketServiceUnitTest`, `SupportTicketCreatedEventListenerUnitTest` (both green), and `./mvnw test-compile -Ptest` (clean, whole project).