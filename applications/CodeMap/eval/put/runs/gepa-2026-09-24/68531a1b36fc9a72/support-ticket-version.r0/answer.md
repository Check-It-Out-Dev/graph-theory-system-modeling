All checks pass: changeset first line is exact, no `LENIENT`/`lenient(`, new test carries `@ExtendWith(MockitoExtension.class)`, and the diff is scoped to exactly what the task needs.

## Summary

- `src/main/java/.../support/ticket/models/SupportTicket.java`: added `@Version @Column(name = "version") private Long version;` for optimistic locking.
- `src/main/resources/db/changelog/2026/09/24-09-2026-add-version-to-support-ticket.sql`: new Liquibase changeset adding `support_ticket.version BIGINT NOT NULL DEFAULT 0`, with rollback.
- `src/main/resources/db/changelog/changelog.xml`: added include for the new changeset.
- `src/test/java/.../unit/service/SupportTicketVersionUnitTest.java`: new `@ExtendWith(MockitoExtension.class)` unit test asserting via reflection that `version` carries `@Version` and is typed `Long`.

Tests run: `SupportTicketVersionUnitTest`, `SupportTicketEntityUnitTest`, `SupportTicketModelsUnitTest`, `SupportTicketServiceUnitTest`, `SupportTicketControllerFullUnitTest` — 391 tests, all passed. `test-compile -Ptest` is green.