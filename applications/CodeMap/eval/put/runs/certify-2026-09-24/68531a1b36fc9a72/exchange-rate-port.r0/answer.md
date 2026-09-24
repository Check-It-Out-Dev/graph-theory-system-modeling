Both checks pass: MockitoExtension present in both new test classes, no code outside the `adapter` package imports an `*Adapter` class, and no `lenient`/`LENIENT` usage. All rule checks are satisfied.

## Summary

- `src/main/java/com/sm/instagram/platform/currency/port/ExchangeRatePort.java` — new port, `Optional<BigDecimal> rateToPln(String isoCode)`.
- `src/main/java/com/sm/instagram/platform/currency/adapter/FixedExchangeRateProperties.java` — `@ConfigurationProperties(prefix = "currency.fixed-rates")`, `Map<String, BigDecimal> rates`.
- `src/main/java/com/sm/instagram/platform/currency/adapter/FixedExchangeRateAdapter.java` — implements the port; normalizes ISO keys to uppercase once at construction, PLN always forced to `1`.
- `src/main/java/com/sm/instagram/platform/currency/CurrencyConversionService.java` — `toPln(amount, isoCode)`, multiplies by the rate, HALF_UP to 2 decimals, throws `BusinessRuleTranslatableException("error.business.exchange_rate_unavailable", isoCode)` when no rate exists.
- `src/main/resources/messages_en.properties`, `messages_pl.properties` — added the new key in both bundles.
- `src/main/resources/application.yml` — added a `currency.fixed-rates.rates` block (EUR, USD) as the interim configured source.
- New tests: `src/test/java/com/sm/instagram/platform/unit/currency/CurrencyConversionServiceUnitTest.java`, `src/test/java/com/sm/instagram/platform/unit/currency/adapter/FixedExchangeRateAdapterUnitTest.java`.

Ran `./mvnw -q test -Ptest -Dtest=CurrencyConversionServiceUnitTest,FixedExchangeRateAdapterUnitTest` — pass; `./mvnw -q test-compile -Ptest` — green.