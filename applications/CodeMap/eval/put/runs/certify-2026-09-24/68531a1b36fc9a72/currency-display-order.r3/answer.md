This all looks correct and complete.

## Summary

- **`src/main/java/com/sm/instagram/platform/currency/Currency.java`**: added `int displayOrder` mapped to `@Column(name = "display_order", nullable = false)`.
- **`src/main/java/com/sm/instagram/platform/currency/CurrencyDtoOut.java`**: added `int displayOrder` field (Lombok `@Data`/`@Builder` cover the getter).
- **`src/main/java/com/sm/instagram/platform/currency/CurrencyService.java`**: `getDataPagedAndFilteredAsDtos` now sets `displayOrder` on the mapped DTO; `toDto` picks it up automatically via ModelMapper's default field-name matching.
- **`src/main/resources/db/changelog/2026/09/24-09-2026-add-display-order-to-currency.sql`** (new) + include in `changelog.xml`: adds `display_order INT NOT NULL DEFAULT 0` and seeds PLN=1, EUR=2, USD=3 (PLN first; none equal the 0 default).
- Fixed all broken `new Currency(...)`/`new CurrencyDtoOut(...)` all-args constructor call sites in `CurrencyServiceUnitTest.java` and `AddressAndCurrencyUnitTest.java`, adding `displayOrder` assertions where natural.
- Added `src/test/java/com/sm/instagram/platform/currency/CurrencyServiceMappingUnitTest.java` (new, `@ExtendWith(MockitoExtension.class)`, no lenient/any()) covering `toDto` and `getDataPagedAndFilteredAsDtos` propagating `displayOrder`.

Tests run: `CurrencyServiceUnitTest`, `CurrencyServiceMappingUnitTest`, `AddressAndCurrencyUnitTest` — all passed. `mvn test-compile -Ptest` — green.

**Needs your attention:** while iterating I first created a test file at the wrong path (`src/test/java/com/sm/instagram/platform/unit/service/CurrencyServiceMappingUnitTest.java`, since `CurrencyService`'s constructor is `protected` and only callable from its own package) and moved the real test to `src/test/java/com/sm/instagram/platform/currency/`. File deletion (`rm`, `git rm`, `mv`, `os.remove`) is blocked in this session, so I could only empty that stray file rather than remove it — please delete `src/test/java/com/sm/instagram/platform/unit/service/CurrencyServiceMappingUnitTest.java` (now 0 bytes, untracked, harmless to the build) yourself.