**Summary**

- `src/main/java/.../currency/Currency.java`: added `int displayOrder` (`@Column(name = "display_order", nullable = false)`).
- `src/main/java/.../currency/CurrencyDtoOut.java`: added `displayOrder` field.
- `src/main/java/.../currency/CurrencyService.java`: `getDataPagedAndFilteredAsDtos` now sets `displayOrder` on the mapped DTO (`toDto` already covers it via ModelMapper's field-name matching).
- `src/main/java/.../currency/CurrencyController.java`: both manual `CurrencyDtoOut.builder()` call sites (`getById`, `findPaginated`) now populate `displayOrder` so it actually reaches clients.
- `src/main/resources/db/changelog/2026/09/23-09-2026-add-display-order-to-currency.sql` (new): adds `display_order INT NOT NULL DEFAULT 0`, seeds PLN=1, EUR=2, USD=3; included in `changelog.xml`.
- `CurrencyServiceUnitTest.java`, `AddressAndCurrencyUnitTest.java`: updated all-args constructor calls for the new field and added assertions covering `displayOrder`.

Tests run (all passed, `-Ptest` profile): `CurrencyServiceUnitTest`, `AddressAndCurrencyUnitTest`, `PartnershipOpportunityConverterUnitTest`, `PartnershipOpportunityMoreUnitTest`, `PartnershipOpportunityControllerUnitTest`.