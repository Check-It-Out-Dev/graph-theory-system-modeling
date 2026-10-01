I already confirmed `test-compile -Ptest` succeeds earlier in this session (before the unrelated permission denial just now), and the targeted test run with `BUILD SUCCESS` also implies compilation succeeded. No further action needed.

## Summary

- `TicketStatsDtoOut.java` (new): DTO with `countsByStatus` map and `total`.
- `SupportTicketRepository.java`: added `countByStatusAndCreatedTimeBetween(TicketStatus, LocalDateTime, LocalDateTime)`.
- `SupportTicketService.java`: added `getTicketStats(LocalDate from, LocalDate to)` — admin-only (same `InsufficientPermissionsException` as `updateTicketStatus`), rejects `from > to` via `ValidationTranslatableException("error.validation.invalid_date_range")`, counts every `TicketStatus` with `start = from.atStartOfDay()` / `end = to.plusDays(1).atStartOfDay()`.
- `SupportTicketController.java`: added `GET /support/ticket/admin/stats?from=&to=`, guarded by `@PreAuthorize("isAuthenticated() && hasAuthority('ADMIN')")` and the class-level `@RateLimit`.
- `messages_en.properties` / `messages_pl.properties`: added `error.validation.invalid_date_range` key to both bundles.
- New tests: `SupportTicketServiceStatsUnitTest.java` and `SupportTicketControllerStatsUnitTest.java` (both `@ExtendWith(MockitoExtension.class)`, no lenient/`LENIENT`).

Tests run: `SupportTicketServiceStatsUnitTest`, `SupportTicketControllerStatsUnitTest`, `SupportTicketServiceUnitTest`, `SupportTicketControllerFullUnitTest` — 167 tests, all passed. `test-compile -Ptest` was green.