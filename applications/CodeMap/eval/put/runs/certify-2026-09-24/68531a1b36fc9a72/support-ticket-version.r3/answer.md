All four changed/added files are needed for this task; no unrelated changes.

## Summary
- `src/main/java/.../support/ticket/models/SupportTicket.java`: added `@Version @Column(name = "version") private Long version;` for optimistic locking.
- `src/main/resources/db/changelog/2026/09/24-09-2026-add-version-to-support-ticket.sql`: new changeset adding `support_ticket.version BIGINT DEFAULT 0 NOT NULL`, with rollback.
- `src/main/resources/db/changelog/changelog.xml`: added the new include at the end.
- `src/test/java/.../unit/service/SupportTicketVersionUnitTest.java`: new `@ExtendWith(MockitoExtension.class)` test asserting the `version` field carries `@Version` and is typed `Long`.

Update paths (`updateTicketStatus`, `addCustomerResponse`, `addAdminResponse`, etc.) already `findById`/mutate/`save` the same managed entity, so Hibernate enforces the optimistic-lock check with no other code changes needed; `OptimisticLockException` is already handled by the existing `BusinessExceptionHandler`.

Tests run: `SupportTicketVersionUnitTest`, `SupportTicketServiceUnitTest`, `SupportTicketEntityUnitTest`, `SupportTicketModelsUnitTest`, `SupportTicketControllerFullUnitTest` — all passed. `mvn test-compile -Ptest` — green.