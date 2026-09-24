Everything checks out. Summary of the change:

- `Currency.java`: added `displayOrder` (int, `@Column(name = "display_order", nullable = false)`).
- `CurrencyDtoOut.java`: added `displayOrder` field.
- `CurrencyService.java`: `getDataPagedAndFilteredAsDtos` now sets `displayOrder` (`toDto` already covered via ModelMapper's name-based mapping).
- Also filled `displayOrder` in every other Currency→CurrencyDtoOut mapping site: `CurrencyController` (`getById`, `findPaginated`), `AppliedOpportunityService`, `PartnershipOpportunityService`, `PartnershipOpportunityController`.
- New Liquibase changeset `db/changelog/2026/09/24-09-2026-add-display-order-to-currency.sql` adds `display_order INT NOT NULL DEFAULT 0` and seeds PLN=1, EUR=2, USD=3; included in `changelog.xml`.
- New test `CurrencyServiceDisplayOrderUnitTest.java` covers both `CurrencyService` mapping paths and the entity's zero default.
- Updated existing tests (`CurrencyServiceUnitTest`, `AddressAndCurrencyUnitTest`, `PartnershipOpportunityControllerUnitTest`) for the new constructor arity/field.

Ran: `CurrencyServiceDisplayOrderUnitTest`, `CurrencyServiceUnitTest`, `AddressAndCurrencyUnitTest`, `PartnershipOpportunityControllerUnitTest`, `PartnershipOpportunityConverterUnitTest`, `PartnershipOpportunityMoreUnitTest` — all pass; `mvnw test-compile -Ptest` is green.