All checks pass: `@ExtendWith(MockitoExtension.class)` present, no `LENIENT`/`lenient(`.

## Summary

- `src/main/java/.../support/ticket/models/SupportTicket.java`: added `@Version @Column(name = "version") private Long version;` field for optimistic locking.
- `src/main/resources/db/changelog/2026/09/24-09-2026-add-version-to-support-ticket.sql`: new changeset adding `support_ticket.version BIGINT NOT NULL DEFAULT 0`, with rollback.
- `src/main/resources/db/changelog/changelog.xml`: added include for the new changeset.
- `src/test/java/.../unit/service/SupportTicketVersionUnitTest.java`: new test asserting via reflection that `version` carries `@Version` and is typed `Long`, maps to the `version` column, and behaves correctly before/after being set.

Tests run: `SupportTicketVersionUnitTest` (4/4 passed), plus existing `SupportTicketServiceUnitTest`, `SupportTicketEntityUnitTest`, `SupportTicketModelsUnitTest`, `SupportTicketControllerFullUnitTest` (all 0 failures/errors). `./mvnw test-compile -Ptest` succeeded.