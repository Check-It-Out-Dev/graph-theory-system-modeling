All tests pass.

Added a swappable exchange-rate conversion feature under `currency/`:
- `currency/port/ExchangeRatePort.java`: port interface, `Optional<BigDecimal> rateToPln(String isoCode)`.
- `currency/adapter/FixedExchangeRateProperties.java`: `@ConfigurationProperties(prefix = "currency.fixed-rates")` with `Map<String, BigDecimal> rates`.
- `currency/adapter/FixedExchangeRateAdapter.java`: implements the port from config, case-insensitive lookup, PLN hardcoded to 1.
- `currency/CurrencyConversionService.java`: `toPln(amount, isoCode)`, multiplies and rounds HALF_UP to 2 decimals; throws `BusinessRuleTranslatableException` with key `error.business.exchange_rate_unavailable` when no rate is found.
- Added the i18n key to `messages_en.properties` and `messages_pl.properties`.
- Added default fixed rates (USD/EUR/GBP) to `application.yml` under `currency.fixed-rates.rates`.
- New tests: `CurrencyConversionServiceUnitTest`, `FixedExchangeRateAdapterUnitTest`.

Ran `./mvnw test -Ptest -Dtest=CurrencyConversionServiceUnitTest,FixedExchangeRateAdapterUnitTest,CurrencyServiceUnitTest` — 47 tests, 0 failures, BUILD SUCCESS.