Compiles cleanly with no errors. All checks pass.

## Summary

- `dtos/TicketStatsDtoOut.java` (new): `countsByStatus: Map<TicketStatus, Long>` and `total: long`.
- `repositories/SupportTicketRepository.java`: added `countByStatusAndCreatedTimeBetween(TicketStatus, LocalDateTime, LocalDateTime)`.
- `services/SupportTicketService.java`: added `getTicketStats(LocalDate from, LocalDate to)` — admin-only (throws `InsufficientPermissionsException` with the same message key/args pattern as `updateTicketStatus`), validates `from <= to` (else `ValidationTranslatableException("error.validation.invalid_date_range")`), sums counts per `TicketStatus` for `[from.atStartOfDay(), to.plusDays(1).atStartOfDay())`.
- `SupportTicketController.java`: added `GET /support/ticket/admin/stats?from=&to=`, `@PreAuthorize("isAuthenticated() && hasAuthority('ADMIN')")`; rate limiting inherited from the existing class-level `@RateLimit`.
- `messages_en.properties` / `messages_pl.properties`: added `error.validation.invalid_date_range`.
- `SupportTicketServiceUnitTest.java`: added `GetTicketStatsTests` nested class (3 tests: happy path with all statuses present, non-admin rejection, invalid range rejection).

Ran `./mvnw -q test -Ptest -Dtest=SupportTicketServiceUnitTest -Dsurefire.failIfNoSpecifiedTests=false` — all tests pass, including the 3 new ones. Also ran `mvnw -q -Ptest compile test-compile` — builds clean.