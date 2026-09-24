Compiles cleanly and tests pass.

## Summary

- `dtos/TicketStatsDtoOut.java` (new): `Map<TicketStatus, Long> countsByStatus` + `long total`.
- `repositories/SupportTicketRepository.java`: added `countByStatusAndCreatedTimeBetween(TicketStatus, LocalDateTime, LocalDateTime)`.
- `services/SupportTicketService.java`: added `getTicketStats(LocalDate from, LocalDate to)` — admin-only (throws `InsufficientPermissionsException` with the same `error.auth.insufficient_permissions` key/args pattern as `updateTicketStatus`), validates `from <= to` (else `ValidationTranslatableException` with `error.validation.invalid_date_range`), sums per-status counts (every `TicketStatus` present) using `start = from.atStartOfDay()`, `end = to.plusDays(1).atStartOfDay()`.
- `SupportTicketController.java`: added `GET /support/ticket/admin/stats?from=&to=` with `@PreAuthorize("isAuthenticated() && hasAuthority('ADMIN')")` (rate-limited via the existing class-level `@RateLimit`).
- `messages_en.properties` / `messages_pl.properties`: added `error.validation.invalid_date_range` key in both.
- `SupportTicketServiceUnitTest.java`: added `GetTicketStatsTests` nested class (deny non-admin, reject invalid range, correct per-status counts and total).

Ran `./mvnw -q test -Ptest -Dtest=SupportTicketServiceUnitTest -Dsurefire.failIfNoSpecifiedTests=false` — all tests passed (quiet mode showed only GDPR log lines, no failures). Also ran `./mvnw -q compile` — clean.