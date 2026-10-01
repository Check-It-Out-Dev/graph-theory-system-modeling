All the intended files. This matches the scope of the task exactly.

## Summary

- `Currency.java`: added `int displayOrder` mapped to a new `display_order` column.
- `CurrencyDtoOut.java`: added `displayOrder`.
- `CurrencyService.java`: `getDataPagedAndFilteredAsDtos` now copies `displayOrder`; `toDto` picks it up via ModelMapper's existing field-name matching.
- `CurrencyController.java`: both manual DTO builders (`getById`, `findPaginated`) now set `displayOrder` too, so the actual admin endpoints return it.
- New changeset `db/changelog/2026/09/24-09-2026-add-display-order-to-currency.sql` (included last in `changelog.xml`): adds the column and sets PLN=1, EUR=2, USD=3 for the three existing rows.
- Updated existing `Currency`/`CurrencyDtoOut` all-args constructor calls in `CurrencyServiceUnitTest.java` and `AddressAndCurrencyUnitTest.java` for the new field.
- New `CurrencyDisplayOrderUnitTest.java` (Mockito, no Spring context) verifying both `toDto` and `getDataPagedAndFilteredAsDtos` carry `displayOrder`, and that PLN's lower value sorts it first.

Tests run: `CurrencyDisplayOrderUnitTest`, `CurrencyServiceUnitTest`, `AddressAndCurrencyUnitTest`, `PartnershipOpportunityMoreUnitTest`, `PartnershipOpportunityConverterUnitTest`, `PartnershipOpportunityControllerUnitTest` — all passing.