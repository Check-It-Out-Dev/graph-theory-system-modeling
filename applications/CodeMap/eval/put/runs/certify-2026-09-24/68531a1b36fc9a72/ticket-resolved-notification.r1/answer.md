All changes are scoped to the support ticket feature. No commit made, per instructions.

**Summary of changes:**
- Added `support/ticket/event/SupportTicketResolvedEvent.java` and `SupportTicketResolvedEventListener.java` (`@TransactionalEventListener(AFTER_COMMIT)`, catches/logs failures).
- Added `support/ticket/port/TicketNotificationPort.java` and `support/ticket/adapter/EmailTicketNotificationAdapter.java` (uses `EmailService#sendEmail`, subject contains ticket reference).
- `SupportTicketService.java`: constructor now takes `ApplicationEventPublisher`; `updateTicketStatus` publishes `SupportTicketResolvedEvent` only when the new status is RESOLVED.
- `SupportTicketServiceUnitTest.java`: added `ApplicationEventPublisher` mock, asserted event payload on RESOLVED, and verified no publish on CLOSED/IN_PROGRESS transitions.
- New `SupportTicketResolvedEventListenerUnitTest.java` and `EmailTicketNotificationAdapterUnitTest.java`.

Tests run: `SupportTicketServiceUnitTest` (all nested classes, incl. `StatusTransitionTests` 6/6), `SupportTicketResolvedEventListenerUnitTest` (2/2), `EmailTicketNotificationAdapterUnitTest` (1/1) — all passed. `mvn test-compile -Ptest` is green.