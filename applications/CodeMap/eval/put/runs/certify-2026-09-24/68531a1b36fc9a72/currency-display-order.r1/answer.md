All checks pass: the changeset header is exact, has a changeset line and rollback, no LENIENT/lenient usage, `@ExtendWith(MockitoExtension.class)` is present. The seeded values (PLN=1, EUR=2, USD=3) all differ from each other and from the column default (0).

## Summary

- `Currency.java`: added `private int displayOrder` mapped to a new `display_order` column.
- `CurrencyDtoOut.java`: added `private int displayOrder`.
- `CurrencyService.java`: `getDataPagedAndFilteredAsDtos` now sets `displayOrder` on each mapped DTO (`toDto` already carries it via ModelMapper's automatic property matching, verified by test).
- `CurrencyController.java`, `PartnershipOpportunityController.java`, `PartnershipOpportunityService.java`, `AppliedOpportunityService.java`: added `.displayOrder(...)` to every other manual `CurrencyDtoOut.builder()` mapping site found via the dependents query.
- New changeset `src/main/resources/db/changelog/2026/09/24-09-2026-add-display-order-to-currency.sql` (+ include in `changelog.xml`): adds `display_order INT NOT NULL DEFAULT 0`, seeds PLN=1, EUR=2, USD=3 so PLN sorts first.
- `CurrencyServiceUnitTest.java`: added `@ExtendWith(MockitoExtension.class)`, updated all 5-arg `Currency(...)` calls for the new constructor arg, added a `CurrencyServiceMappingTests` nested class testing both `toDto` and `getDataPagedAndFilteredAsDtos` map `displayOrder`.
- `AddressAndCurrencyUnitTest.java`: updated the `CurrencyDtoOut` all-args constructor call and assertion for the new field.

Tests run: `CurrencyServiceUnitTest`, `AddressAndCurrencyUnitTest`, `PartnershipOpportunityControllerUnitTest`, `PartnershipOpportunityConverterUnitTest`, `PartnershipOpportunityMoreUnitTest`, `AppliedOpportunityServiceUnitTest` — all green; `mvnw test-compile -Ptest` — green.