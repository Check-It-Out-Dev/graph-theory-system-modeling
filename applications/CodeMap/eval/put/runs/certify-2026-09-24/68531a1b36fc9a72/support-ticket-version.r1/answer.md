All checks pass: changeset header is exact, no existing changeset was modified, and only the intended files changed.

## Summary

- `src/main/java/com/sm/instagram/platform/support/ticket/models/SupportTicket.java`: added `@Version private Long version;` field for JPA optimistic locking (exemplar: `PartnershipOpportunity.java`).
- `src/main/resources/db/changelog/2026/09/24-09-2026-add-version-to-support-ticket.sql`: new changeset adding `version BIGINT DEFAULT 0 NOT NULL` to `support_ticket`, with rollback.
- `src/main/resources/db/changelog/changelog.xml`: added include for the new changeset.
- `src/test/java/com/sm/instagram/platform/support/ticket/models/SupportTicketVersionUnitTest.java`: new `@ExtendWith(MockitoExtension.class)` test asserting `@Version`/`Long` by reflection and that concurrently loaded copies diverge in version after one write.
- No new translatable exception was needed — `OptimisticLockingFailureException` is already handled globally by `BusinessExceptionHandler` with key `error.database.concurrent_modification`, present in both `messages_en.properties` and `messages_pl.properties`.

Tests run: `SupportTicketVersionUnitTest`, `SupportTicketServiceUnitTest`, `SupportTicketEntityUnitTest`, `SupportTicketModelsUnitTest` — all passed. `./mvnw -q test-compile -Ptest` — green.