All green (no failures reported, `-q` suppresses the BUILD SUCCESS summary but no errors surfaced).

Note: `CurrencyServiceMappingUnitTest.java` lives under `unit/service/` on disk but declares package `com.sm.instagram.platform.currency` (needed to reach `CurrencyService`'s protected constructor); this compiles and runs correctly under Maven.

## Summary
- `Currency.java`: added `@Column(name = "display_order", nullable = false) private int displayOrder;`.
- `CurrencyDtoOut.java`: added `private int displayOrder;`.
- `CurrencyService.java`: `getDataPagedAndFilteredAsDtos` now sets `displayOrder` on the DTO (`toDto` already covers it via ModelMapper's matching field names).
- `CurrencyController.java`: both manual `CurrencyDtoOut.builder()` mappings (`getById`, `findPaginated`) now include `.displayOrder(...)` so clients actually receive it.
- New Liquibase changeset `src/main/resources/db/changelog/2026/09/24-09-2026-add-display-order-to-currency.sql` adds the `display_order` column (default 0) and seeds PLN=0, EUR=1, USD=2; included in `changelog.xml`.
- Updated existing tests (`CurrencyServiceUnitTest`, `AddressAndCurrencyUnitTest`) for the new constructor arity and added `displayOrder` assertions.
- Added `CurrencyServiceMappingUnitTest` (Mockito, strict) verifying `toDto` and `getDataPagedAndFilteredAsDtos` both carry `displayOrder`.
- Tests run: `CurrencyServiceUnitTest`, `CurrencyServiceMappingUnitTest`, `AddressAndCurrencyUnitTest`, `PartnershipOpportunityControllerUnitTest` — all passed; `test-compile -Ptest` is green.