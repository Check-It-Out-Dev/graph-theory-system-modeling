No matches for failure/error markers — build succeeded with all tests green (Maven `-q` only prints output on failure, and none of the failure keywords appear in the captured log).

## Summary

- `SupportTicket.java` (models): added `@Version private Long version;` (JPA optimistic locking).
- `db/changelog/2026/09/24-09-2026-add-version-to-support-ticket.sql`: new changeset adding `version BIGINT DEFAULT 0 NOT NULL` to `support_ticket`, with rollback.
- `changelog.xml`: included the new changeset with a September 2026 comment.
- New test `SupportTicketVersionUnitTest.java`: verifies `version` is `Long`, annotated `@Version`, defaults to `null`, and that `updateTicketStatus`/`addCustomerResponseAndReturnDto` propagate `ObjectOptimisticLockingFailureException` (already translated to HTTP 409 by the existing global `BusinessExceptionHandler`, so no service/controller change was needed).

Tests run: `SupportTicketVersionUnitTest`, `SupportTicketServiceUnitTest`, `SupportTicketEntityUnitTest`, `SupportTicketModelsUnitTest`, `SupportTicketControllerFullUnitTest` — all passed.