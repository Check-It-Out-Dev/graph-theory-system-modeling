This is the existing `@Autowired` explicit constructor pattern (single constructor, `@Autowired` optional but pre-existing style kept — this is constructor injection, compliant with `constructor_injection` rule, not field/setter injection). All rule checks pass.

## Summary

- `src/main/java/.../support/ticket/event/SupportTicketCreatedEvent.java` — new event carrying ticket id, contact email, reference, subject, language, status token.
- `src/main/java/.../support/ticket/event/SupportTicketCreatedEventListener.java` — new `@TransactionalEventListener(phase = AFTER_COMMIT)` that sends the confirmation via `EmailService`, catching/logging failures.
- `src/main/java/.../support/ticket/services/SupportTicketService.java` — `createTicket` now publishes `SupportTicketCreatedEvent` via constructor-injected `ApplicationEventPublisher` instead of calling `EmailService` directly.
- `src/test/java/.../support/ticket/event/SupportTicketCreatedEventListenerUnitTest.java` — new unit test (send path + swallowed-failure path).
- `src/test/java/.../unit/service/SupportTicketServiceUnitTest.java` — added `createTicket` tests asserting the published event's values and that `EmailService` is never called directly.
- `src/test/java/.../integration/service/supportticket/SupportTicketService_Create_IntegrationTest.java` — updated the email-confirmation test to assert no direct `EmailService` interaction (AFTER_COMMIT can't fire in the rollback-based test transaction, matching the existing convention in `NotificationService_AccountActivation_IntegrationTest`).

Tests run: `SupportTicketServiceUnitTest` (all pass, including new `CreateTicketTests`), `SupportTicketCreatedEventListenerUnitTest` (2/2 pass), and `mvn test-compile -Ptest` (clean, no errors).