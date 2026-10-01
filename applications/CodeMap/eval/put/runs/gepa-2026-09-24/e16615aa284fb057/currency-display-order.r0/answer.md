All checks pass: no lenient/LENIENT/computed `any()` in the new test, no existing changeset was modified, and the new changeset + include were added cleanly. This change doesn't involve translatable errors, ports/adapters, schedulers, or event listeners, so those rules don't apply here.

## Summary

- `Currency.java`: added `private int displayOrder;` (`display_order` column).
- `CurrencyDtoOut.java`: added `private int displayOrder;`.
- `CurrencyService.java`: `getDataPagedAndFilteredAsDtos` now sets `displayOrder`; `toDto` picks it up automatically via ModelMapper's field-name matching.
- `CurrencyController.java`, `PartnershipOpportunityService.java`, `PartnershipOpportunityController.java`, `AppliedOpportunityService.java`: the other four places that build `CurrencyDtoOut` from a `Currency` now also set `displayOrder` (dto_naming rule — no mapping left silently defaulting).
- New Liquibase changeset `2026/09/24-09-2026-add-display-order-to-currency.sql` adds the column (default 0) and backfills PLN=1, EUR=2, USD=3; included in `changelog.xml`.
- Updated existing tests (`CurrencyServiceUnitTest.java`, `AddressAndCurrencyUnitTest.java`) whose all-args constructor calls broke due to the new field.
- Added `src/test/java/com/sm/instagram/platform/currency/CurrencyServiceDisplayOrderUnitTest.java` covering `toDto`, the paged mapping (PLN/EUR/USD order preserved), and the empty-page case.

Tests run: `CurrencyServiceDisplayOrderUnitTest`, `CurrencyServiceUnitTest`, `AddressAndCurrencyUnitTest` (105 tests) and `PartnershipOpportunityConverterUnitTest`/`ControllerUnitTest`/`MoreUnitTest` (260 tests) — all passed; `test-compile -Ptest` is green.