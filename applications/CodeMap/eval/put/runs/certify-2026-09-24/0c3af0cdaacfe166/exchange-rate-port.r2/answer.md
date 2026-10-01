All 6 new tests pass.

Added a port/adapter pair for exchange rates and a service that converts to PLN:
- `currency/port/ExchangeRatePort.java` — new port interface, `rateToPln(String isoCode)`.
- `currency/adapter/FixedExchangeRateProperties.java` — `@ConfigurationProperties(prefix = "currency.fixed-rates")` holding `Map<String, BigDecimal> rates`.
- `currency/adapter/FixedExchangeRateAdapter.java` — implements the port from properties; case-insensitive match, PLN hard-coded to 1.
- `currency/CurrencyConversionService.java` — `toPln(amount, isoCode)`, rounds HALF_UP to 2 decimals, throws `BusinessRuleTranslatableException` with key `error.business.exchange_rate_unavailable` when no rate exists.
- Added the message key to `messages_en.properties` and `messages_pl.properties`.
- Added sample `currency.fixed-rates.rates` (USD/EUR/GBP) to `application.yml`.
- Added `FixedExchangeRateAdapterUnitTest` and `CurrencyConversionServiceUnitTest` under `src/test/.../unit/currency/`; ran via `./mvnw test -Ptest -Dtest=CurrencyConversionServiceUnitTest,FixedExchangeRateAdapterUnitTest` — 6/6 passed.