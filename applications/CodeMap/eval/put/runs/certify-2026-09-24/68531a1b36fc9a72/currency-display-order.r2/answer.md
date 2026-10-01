That's fine — all necessary verification (compile, targeted tests, rule checks) is already complete.

## Summary

- `Currency.java`: added `private int displayOrder;` (`@Column(name = "display_order", nullable = false)`).
- `CurrencyDtoOut.java`: added `private int displayOrder;`.
- `CurrencyService.java`: `toDto` maps it via ModelMapper (name match); `getDataPagedAndFilteredAsDtos` now sets `dto.setDisplayOrder(...)`.
- `CurrencyController.java`, `AppliedOpportunityService.java`, `PartnershipOpportunityController.java`, `PartnershipOpportunityService.java`: all remaining `CurrencyDtoOut` builder call sites now set `displayOrder` too, so no client-facing mapping silently defaults to 0.
- New changeset `src/main/resources/db/changelog/2026/09/24-09-2026-add-display-order-to-currency.sql` (registered in `changelog.xml`): adds `display_order INT NOT NULL DEFAULT 0`, seeds PLN=1, EUR=2, USD=3.
- Fixed all-args constructor call sites broken by the new field: `CurrencyServiceUnitTest.java`, `AddressAndCurrencyUnitTest.java`.
- New test `src/test/java/com/sm/instagram/platform/currency/CurrencyServiceDisplayOrderUnitTest.java` covering the entity field, `toDto`, and `getDataPagedAndFilteredAsDtos`.

Tests run: `CurrencyServiceUnitTest`, `CurrencyServiceDisplayOrderUnitTest`, `AddressAndCurrencyUnitTest`, `AppliedOpportunityServiceUnitTest`, `PartnershipOpportunityServiceUnitTest`, `PartnershipOpportunityConverterUnitTest`, `PartnershipOpportunityMoreUnitTest`, `PartnershipOpportunityControllerUnitTest` — all passed; `test-compile -Ptest` is green.