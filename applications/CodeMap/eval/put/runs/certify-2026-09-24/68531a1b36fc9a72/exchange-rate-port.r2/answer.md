All checks pass — no `@Autowired` added, no lenient/LENIENT, no cross-adapter imports outside the adapter package, message keys present in both bundles, tests annotated correctly and green, and `test-compile -Ptest` is clean.

## Summary

- `src/main/java/.../currency/port/ExchangeRatePort.java` — new port, `Optional<BigDecimal> rateToPln(String isoCode)`.
- `src/main/java/.../currency/adapter/FixedExchangeRateProperties.java` — `@ConfigurationProperties(prefix = "currency.fixed-rates")`, `Map<String, BigDecimal> rates`.
- `src/main/java/.../currency/adapter/FixedExchangeRateAdapter.java` — implements the port; normalizes ISO keys to upper case once in the constructor, `PLN` fixed at `1`.
- `src/main/java/.../currency/CurrencyConversionService.java` — constructor-injects the port, `toPln` multiplies and rounds HALF_UP to 2 decimals, throws `BusinessRuleTranslatableException("error.business.exchange_rate_unavailable", isoCode)` when no rate is found.
- `messages_en.properties` / `messages_pl.properties` — added `error.business.exchange_rate_unavailable` to both bundles.
- `application.yml` — added an example `currency.fixed-rates.rates` block (USD/EUR/GBP).
- New tests: `unit/currency/adapter/FixedExchangeRateAdapterUnitTest.java`, `unit/currency/CurrencyConversionServiceUnitTest.java`.

Tests run: `FixedExchangeRateAdapterUnitTest`, `CurrencyConversionServiceUnitTest` — passed. `mvnw test-compile -Ptest` — green.