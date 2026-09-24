All 76 tests pass, including the 2 new version tests.

**Summary:**
- `src/main/java/com/sm/instagram/platform/support/ticket/models/SupportTicket.java`: added `@Version private Long version;` field for optimistic locking.
- `src/main/resources/db/changelog/2026/09/23-09-2026-add-version-to-support-ticket.sql`: new Liquibase changeset adding `version BIGINT DEFAULT 0 NOT NULL` to `support_ticket`.
- `src/main/resources/db/changelog/changelog.xml`: included the new changeset.
- `src/test/java/com/sm/instagram/platform/unit/service/SupportTicketEntityUnitTest.java`: added `VersionTests` nested class covering the default null value and getter/setter.

Tests run: `./mvnw test -Ptest -Dtest=SupportTicketEntityUnitTest,SupportTicketModelsUnitTest,SupportTicketServiceUnitTest` — all pass (76/76 in the entity test class, no failures elsewhere).