All 6 new tests pass. No existing classes were modified besides config/messages, so there's nothing else to re-run.

## Summary

Added a currency-to-PLN conversion behind a swappable port, with rates fixed in configuration for now:

- `src/main/java/com/sm/instagram/platform/currency/port/ExchangeRatePort.java` — new port, `Optional<BigDecimal> rateToPln(String isoCode)`.
- `src/main/java/com/sm/instagram/platform/currency/adapter/FixedExchangeRateProperties.java` — `@ConfigurationProperties(prefix = "currency.fixed-rates")` holding `Map<String, BigDecimal> rates`.
- `src/main/java/com/sm/instagram/platform/currency/adapter/FixedExchangeRateAdapter.java` — implements the port from the fixed properties; case-insensitive match, `PLN` always `1`.
- `src/main/java/com/sm/instagram/platform/currency/CurrencyConversionService.java` — `toPln(amount, isoCode)`, multiplies by rate, HALF_UP to 2 decimals, throws `BusinessRuleTranslatableException("error.business.exchange_rate_unavailable", isoCode)` when no rate exists.
- `src/main/resources/messages_en.properties` and `messages_pl.properties` — added the new key next to the other `error.business.*` entries.
- `src/main/resources/application.yml` — added a sample `currency.fixed-rates.rates` block (PLN/USD/EUR/GBP).
- New tests: `src/test/java/.../unit/currency/FixedExchangeRateAdapterUnitTest.java`, `src/test/java/.../unit/currency/CurrencyConversionServiceUnitTest.java`.

Ran `./mvnw test -Ptest -Dtest=FixedExchangeRateAdapterUnitTest,CurrencyConversionServiceUnitTest -Dsurefire.failIfNoSpecifiedTests=false` — 6 tests, all passed, BUILD SUCCESS.