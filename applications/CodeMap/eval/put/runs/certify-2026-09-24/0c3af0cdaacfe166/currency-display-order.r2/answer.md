All green.

## Summary

- `Currency.java`: added `int displayOrder` mapped to a new `display_order` column.
- `CurrencyDtoOut.java`: added `displayOrder` field (via `@Data` → `getDisplayOrder()`).
- `CurrencyService.java`: `getDataPagedAndFilteredAsDtos` now copies `displayOrder` onto the DTO; `toDto` picks it up automatically via `modelMapper` (matching field names).
- New Liquibase changeset `src/main/resources/db/changelog/2026/09/23-09-2026-currency-display-order.sql` adds `display_order INT NOT NULL DEFAULT 0` and seeds PLN=0, EUR=1, USD=2; included in `changelog.xml`.
- Updated `CurrencyServiceUnitTest.java` and `AddressAndCurrencyUnitTest.java` to pass the new constructor arg to `Currency`/`CurrencyDtoOut`.
- Added `CurrencyServiceDisplayOrderUnitTest.java` (Mockito, no Spring context) covering `toDto` and `getDataPagedAndFilteredAsDtos` mapping `displayOrder`.

Tests run: `CurrencyServiceUnitTest`, `CurrencyServiceDisplayOrderUnitTest`, `AddressAndCurrencyUnitTest`, plus `PartnershipOpportunityControllerUnitTest`, `PartnershipOpportunityMoreUnitTest`, `PartnershipOpportunityConverterUnitTest` (all touching `Currency`) — 362+104 tests, all passing; full `test-compile` also succeeds.