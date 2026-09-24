Changes are scoped exactly as needed.

- `src/main/java/com/sm/instagram/platform/support/ticket/models/SupportTicket.java`: added `@Version private Long version;` (exemplar: `User.java`'s optimistic-locking field), enabling JPA optimistic locking on concurrent admin/customer edits.
- `src/main/resources/db/changelog/2026/09/24-09-2026-add-version-to-support-ticket.sql`: new changeset adding `support_ticket.version BIGINT NOT NULL DEFAULT 0` (exemplar: `20-12-2025-add-version-to-user.sql`), with rollback.
- `src/main/resources/db/changelog/changelog.xml`: included the new changeset at the end.

Tests run: `./mvnw -q test-compile -Ptest` — green. `./mvnw -q test -Ptest -Dtest=SupportTicketModelsUnitTest,SupportTicketEntityUnitTest,SupportTicketServiceUnitTest,SupportTicketControllerFullUnitTest -Dsurefire.failIfNoSpecifiedTests=false` — all passed (no failures/errors). No new unit test class added since this is a persistence-only field with Lombok-generated accessors and no new logic, consistent with the `User`/`AppliedOpportunity`/`Content` version-field precedents.