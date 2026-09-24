Compile and the targeted unit tests (`SupportTicketModelsUnitTest`, `SupportTicketServiceUnitTest`, `SupportTicketEntityUnitTest`, `SupportTicketControllerFullUnitTest`) ran clean with `-q` (no `BUILD FAILURE` / failure output — Maven's quiet mode only prints on failure, and the logged `ERROR`/`GDPR` lines are expected test-scenario output, not failures).

Summary:
- `src/main/java/com/sm/instagram/platform/support/ticket/models/SupportTicket.java`: added `@Version @Column(name = "version") private Long version;` for optimistic locking, following the `AppliedOpportunity` pattern.
- `src/main/resources/db/changelog/2026/09/23-09-2026-add-version-to-support-ticket.sql`: new changeset adding `support_ticket.version BIGINT NOT NULL DEFAULT 0`, with rollback.
- `src/main/resources/db/changelog/changelog.xml`: included the new changeset.
- Tests run: `SupportTicketModelsUnitTest`, `SupportTicketServiceUnitTest`, `SupportTicketEntityUnitTest`, `SupportTicketControllerFullUnitTest` — all passed. Full `mvn compile` also succeeded.