All checks pass. Changes:

- `src/main/java/.../support/ticket/models/SupportTicket.java` — added `@Version @Column(name = "version") private Long version;` field for optimistic locking (exemplar: `PartnershipOpportunity.java`).
- `src/main/resources/db/changelog/2026/09/24-09-2026-add-version-to-support-ticket.sql` — new changeset adding `support_ticket.version BIGINT DEFAULT 0 NOT NULL` with rollback (exemplar: `20-12-2025-add-version-to-partnership-opportunity.sql`).
- `src/main/resources/db/changelog/changelog.xml` — added include for the new changeset.
- `src/test/java/.../support/ticket/models/SupportTicketUnitTest.java` — new unit test asserting default `null` version and set/get round-trip.

Tests run: `SupportTicketUnitTest` (new), `SupportTicketEntityUnitTest`, `SupportTicketModelsUnitTest`, `SupportTicketServiceUnitTest`, `SupportTicketControllerFullUnitTest` via `-Ptest -Dtest=...` — all passed. `test-compile -Ptest` — green.