All checks pass: `@ExtendWith(MockitoExtension.class)` present, no LENIENT/lenient/any(), no adapter imports outside adapter packages, both message bundles updated, controller endpoint has `@PreAuthorize` (class-level `@RateLimit` already covers it), and the DTO/service/repository match the exact names in the interface. Build and targeted tests are green.

## Summary

- `SupportTicketRepository.java`: added `countByStatusAndCreatedTimeBetween(TicketStatus, LocalDateTime, LocalDateTime)`.
- `TicketStatsDtoOut.java` (new): `Map<TicketStatus, Long> countsByStatus` + `long total`.
- `SupportTicketService.java`: added `getTicketStats(LocalDate from, LocalDate to)` — admin-only (same `InsufficientPermissionsException` as `updateTicketStatus`), validates `from`/`to` (`ValidationTranslatableException`, key `error.validation.invalid_date_range`), queries every `TicketStatus` with `start = from.atStartOfDay()`, `end = to.plusDays(1).atStartOfDay()`.
- `SupportTicketController.java`: added `GET /support/ticket/admin/stats?from=&to=` with `@PreAuthorize("isAuthenticated() && hasAuthority('ADMIN')")` (class-level `@RateLimit` already applies).
- `messages_en.properties` / `messages_pl.properties`: added `error.validation.invalid_date_range` in both.
- New tests: `SupportTicketServiceStatsUnitTest.java` (5 tests: happy path, all-zero path, single-day boundary, invalid range, non-admin) and `SupportTicketControllerStatsUnitTest.java` (3 tests: success + two auth-missing paths).

Tests run: `SupportTicketServiceStatsUnitTest` (5/5 pass), `SupportTicketControllerStatsUnitTest` (3/3 pass), plus existing `SupportTicketServiceUnitTest` and `SupportTicketControllerFullUnitTest` (all still pass). `mvn test-compile -Ptest` is green.