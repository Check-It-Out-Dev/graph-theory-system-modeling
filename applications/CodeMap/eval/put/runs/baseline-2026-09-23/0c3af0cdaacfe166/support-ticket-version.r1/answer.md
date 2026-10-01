Build compiles cleanly and all SupportTicket unit tests pass.

## Summary

- `src/main/java/com/sm/instagram/platform/support/ticket/models/SupportTicket.java`: added a `@Version` `private Long version;` field for JPA optimistic locking, following the `AppliedOpportunityContent` pattern.
- `src/main/resources/db/changelog/2026/09/23-09-2026-add-version-to-support-ticket.sql`: new Liquibase changeset adding `support_ticket.version BIGINT NOT NULL DEFAULT 0`, with rollback.
- `src/main/resources/db/changelog/changelog.xml`: included the new changeset with a comment.

Tests run: `SupportTicketEntityUnitTest`, `SupportTicketModelsUnitTest`, `SupportTicketServiceUnitTest`, `SupportTicketControllerFullUnitTest` — all passed (0 failures/errors across all nested test classes). Also ran `mvn compile` — clean build.