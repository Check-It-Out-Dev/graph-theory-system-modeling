All new files are additive; no existing constructors or DTOs changed, so no other tests need updates.

## Summary

- `src/main/java/com/sm/instagram/platform/currency/port/ExchangeRatePort.java` — new port interface, `rateToPln(String)`.
- `src/main/java/com/sm/instagram/platform/currency/adapter/FixedExchangeRateProperties.java` — `@ConfigurationProperties(prefix = "currency.fixed-rates")` with `Map<String, BigDecimal> rates`.
- `src/main/java/com/sm/instagram/platform/currency/adapter/FixedExchangeRateAdapter.java` — implements the port from fixed config, case-insensitive lookup, PLN always 1.
- `src/main/java/com/sm/instagram/platform/currency/CurrencyConversionService.java` — `toPln(amount, isoCode)`, HALF_UP to 2 decimals, throws `BusinessRuleTranslatableException("error.business.exchange_rate_unavailable", isoCode)` when no rate.
- `src/main/resources/messages_en.properties`, `messages_pl.properties` — added `error.business.exchange_rate_unavailable` key in both.
- `src/main/resources/application.yml` — added `currency.fixed-rates.rates` sample config (EUR/USD/GBP).
- `src/test/java/.../unit/currency/adapter/FixedExchangeRateAdapterUnitTest.java` and `.../unit/currency/CurrencyConversionServiceUnitTest.java` — new unit tests.

Ran: `./mvnw -q test-compile -Ptest` (green) and `./mvnw -q test -Ptest -Dtest=FixedExchangeRateAdapterUnitTest,CurrencyConversionServiceUnitTest` (both passed, no failures).