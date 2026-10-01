All changes are in the working tree, not committed, as required.

**Summary:**
- `SupportTicketResolvedEvent.java` (new, `event` pkg): carries ticketId, contactEmail, ticketReference.
- `TicketNotificationPort.java` (new, `port` pkg): `notifyResolved(contactEmail, ticketReference)`.
- `EmailTicketNotificationAdapter.java` (new, `adapter` pkg): implements the port via `EmailService#sendEmail`, subject includes the reference.
- `SupportTicketResolvedEventListener.java` (new, `event` pkg): `@TransactionalEventListener(AFTER_COMMIT)`, catches/logs port failures.
- `SupportTicketService.java`: injects `ApplicationEventPublisher`, publishes `SupportTicketResolvedEvent` in `updateTicketStatus` only when new status is RESOLVED.
- `SupportTicketServiceUnitTest.java`: added `ApplicationEventPublisher` mock; asserts event contents on RESOLVED, `verifyNoInteractions(eventPublisher)` on IN_PROGRESS/CLOSED.
- New tests: `EmailTicketNotificationAdapterUnitTest`, `SupportTicketResolvedEventListenerUnitTest` (happy path + swallowed-failure path).

Tests run: `./mvnw test -Ptest -Dtest=SupportTicketServiceUnitTest,EmailTicketNotificationAdapterUnitTest,SupportTicketResolvedEventListenerUnitTest` → 56/56 passed. `./mvnw test-compile -Ptest` → BUILD SUCCESS.