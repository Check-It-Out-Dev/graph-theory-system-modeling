All changed files are exactly what the task needs. Checks pass: no LENIENT/lenient(), `@ExtendWith(MockitoExtension.class)` present, both message bundles updated, controller has `@PreAuthorize` + `@RateLimit`, service checks admin permission and date range, tests compile and pass.

## Summary

- `dtos/TicketStatsDtoOut.java` (new): `countsByStatus` map and `total` field.
- `repositories/SupportTicketRepository.java`: added `countByStatusAndCreatedTimeBetween(TicketStatus, LocalDateTime, LocalDateTime)`.
- `services/SupportTicketService.java`: added `getTicketStats(LocalDate, LocalDate)` — admin-only (same `InsufficientPermissionsException`/`error.auth.insufficient_permissions` as `updateTicketStatus`), validates `from <= to` (else `ValidationTranslatableException` with `error.validation.invalid_date_range`), counts every `TicketStatus` via `from.atStartOfDay()`/`to.plusDays(1).atStartOfDay()`.
- `SupportTicketController.java`: added `GET /support/ticket/admin/stats` with `@PreAuthorize("isAuthenticated() && hasAuthority('ADMIN')")` and `@RateLimit(profile = RateLimitProfile.STANDARD)`.
- `messages_en.properties` / `messages_pl.properties`: added `error.validation.invalid_date_range` key in both.
- New test: `SupportTicketServiceStatsUnitTest.java` (`@ExtendWith(MockitoExtension.class)`, strict Mockito) — happy path with per-status assertions, all-zero path, invalid range, and non-admin rejection.

Tests run: `SupportTicketServiceStatsUnitTest` (4/4 pass), `SupportTicketServiceUnitTest` and `SupportTicketControllerFullUnitTest` (existing, all pass), `mvnw test-compile -Ptest` green.