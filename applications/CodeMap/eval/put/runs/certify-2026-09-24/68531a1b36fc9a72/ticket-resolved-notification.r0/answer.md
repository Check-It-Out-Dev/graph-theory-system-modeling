All checks pass: AFTER_COMMIT present, both new test files have `@ExtendWith(MockitoExtension.class)`, no LENIENT/lenient usage, no adapter imports outside the adapter package. Everything is consistent with the manual's rules.

## Summary

- `support/ticket/event/SupportTicketResolvedEvent.java`, `support/ticket/event/SupportTicketResolvedEventListener.java` (new): event published on RESOLVED, listener notifies via the port `AFTER_COMMIT`, catching/logging failures.
- `support/ticket/port/TicketNotificationPort.java`, `support/ticket/adapter/EmailTicketNotificationAdapter.java` (new): replaceable notification port with today's e-mail adapter, subject/body built from new i18n keys.
- `messages_en.properties`, `messages_pl.properties`: added `email.ticket_resolved.subject`/`.body` in both bundles.
- `SupportTicketService.java`: constructor now takes `ApplicationEventPublisher`; `updateTicketStatus` publishes `SupportTicketResolvedEvent` only when the new status is RESOLVED.
- `SupportTicketServiceUnitTest.java`: added `ApplicationEventPublisher` mock, captured/asserted the published event on RESOLVED, and verified no event on CLOSED.
- New `SupportTicketResolvedEventListenerUnitTest.java` and `EmailTicketNotificationAdapterUnitTest.java`.

Tests run: `SupportTicketServiceUnitTest` (53 tests across nested classes), `SupportTicketResolvedEventListenerUnitTest` (2), `EmailTicketNotificationAdapterUnitTest` (1) — all passed. `mvnw test-compile -Ptest` green.