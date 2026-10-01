All test classes touched pass.

Summary:
- `Currency.java`: added `private int displayOrder;` mapped to a new `display_order` column.
- `CurrencyDtoOut.java`: added `displayOrder` field.
- `CurrencyService.java`: `getDataPagedAndFilteredAsDtos` now sets `displayOrder` (the `toDto` path picks it up automatically via ModelMapper's field-name mapping).
- `CurrencyController.java`: both manual `CurrencyDtoOut` builders (`getById`, `findPaginated`) now populate `displayOrder`, so the value actually reaches clients.
- Added `src/main/resources/db/changelog/2026/09/23-09-2026-add-display-order-to-currency.sql`, adding the `display_order INT NOT NULL DEFAULT 0` column and seeding PLN=0, EUR=1, USD=2; registered it in `changelog.xml`.
- Updated the `Currency`/`CurrencyDtoOut` all-args constructor call sites and added `displayOrder` assertions in `CurrencyServiceUnitTest.java` and `AddressAndCurrencyUnitTest.java`.

Tests run: `CurrencyServiceUnitTest` (42 tests), `AddressAndCurrencyUnitTest`, `PartnershipOpportunityControllerUnitTest`, `PartnershipOpportunityConverterUnitTest`, `PartnershipOpportunityMoreUnitTest` — all pass (`BUILD SUCCESS`).