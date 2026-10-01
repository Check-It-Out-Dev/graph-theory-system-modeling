The previous run already completed with quiet mode (`-q`), which only prints output on failure — Maven's `-q test` suppresses BUILD SUCCESS/FAILURE banners but surfaces failures. Since no failure/error text appeared, all tests passed. The test target file compiled and ran; the maven build finished without printing failure diagnostics.

## Summary

- `src/main/java/com/sm/instagram/platform/support/ticket/dtos/TicketStatsDtoOut.java` (new): `countsByStatus` map and `total`.
- `src/main/java/com/sm/instagram/platform/support/ticket/repositories/SupportTicketRepository.java`: added `countByStatusAndCreatedTimeBetween`.
- `src/main/java/com/sm/instagram/platform/support/ticket/services/SupportTicketService.java`: added `getTicketStats(LocalDate, LocalDate)` — admin-only (reuses `updateTicketStatus`'s `InsufficientPermissionsException`), throws `ValidationTranslatableException("error.validation.invalid_date_range")` when `from` is after `to`, counts every `TicketStatus`.
- `src/main/java/com/sm/instagram/platform/support/ticket/SupportTicketController.java`: added `GET /support/ticket/admin/stats?from=&to=`, `@PreAuthorize("isAuthenticated() && hasAuthority('ADMIN')")`.
- `src/main/resources/messages_en.properties` and `messages_pl.properties`: added `error.validation.invalid_date_range`.
- `src/test/java/com/sm/instagram/platform/unit/service/SupportTicketServiceUnitTest.java`: added `GetTicketStatsTests` (permission denial, invalid range, per-status counts with all statuses present).

Ran `./mvnw -q test -Ptest -Dtest=SupportTicketServiceUnitTest -Dsurefire.failIfNoSpecifiedTests=false` — all tests passed (no failures reported).