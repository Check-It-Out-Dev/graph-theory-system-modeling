Added the admin ticket-stats endpoint end to end:

- `TicketStatsDtoOut.java` (new) — `countsByStatus` map + `total`.
- `SupportTicketRepository.java` — added `countByStatusAndCreatedTimeBetween`.
- `SupportTicketService.java` — added `getTicketStats(from, to)`: admin-only check (same `InsufficientPermissionsException` as `updateTicketStatus`), validates `from`/`to` via `ValidationTranslatableException("error.validation.invalid_date_range")`, sums counts across all `TicketStatus` values.
- `SupportTicketController.java` — added `GET /support/ticket/admin/stats` with `@PreAuthorize("isAuthenticated() && hasAuthority('ADMIN')")` and `@RateLimit(profile = RateLimitProfile.STANDARD)`.
- `messages_en.properties` / `messages_pl.properties` — added `error.validation.invalid_date_range` key to both bundles.
- New tests: `SupportTicketServiceTicketStatsUnitTest.java` (4 tests: counts per status/total, all-zero period, invalid range, non-admin denial) and `SupportTicketControllerStatsUnitTest.java` (2 tests: success path, missing-auth path).

Tests run: `SupportTicketServiceTicketStatsUnitTest` + `SupportTicketControllerStatsUnitTest` (6/6 pass), plus the full pre-existing `SupportTicketServiceUnitTest` and `SupportTicketControllerFullUnitTest` suites (167/167 pass); `test-compile -Ptest` green.