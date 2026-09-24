Everything checks out against the rules (constructor injection, AFTER_COMMIT with catch/log, event/listener in the feature's `event` package, no `EmailService` call left in the service, tests updated and passing).

## Summary

- Added `com.sm.instagram.platform.support.ticket.event.SupportTicketCreatedEvent` and `SupportTicketCreatedEventListener` (`@TransactionalEventListener(phase = AFTER_COMMIT)`, catches/logs failures) — the listener sends the confirmation via `EmailService#sendTicketCreationConfirmation`.
- `SupportTicketService`: constructor now takes `ApplicationEventPublisher`; `createTicket` publishes `SupportTicketCreatedEvent` instead of calling `EmailService` directly.
- Updated `SupportTicketServiceUnitTest` (added `ApplicationEventPublisher` mock, new `createTicket` test asserting the published event's fields and that `EmailService` is never called synchronously) and added `SupportTicketCreatedEventListenerUnitTest` (send path + swallowed-failure path).
- Updated the integration test that previously asserted a synchronous email send; it now asserts no synchronous `EmailService` interaction (matches existing convention for AFTER_COMMIT listeners not firing in `@Transactional` test rollback).

Tests run: `SupportTicketServiceUnitTest`, `SupportTicketCreatedEventListenerUnitTest` — passed. `./mvnw test-compile -Ptest` — passed.