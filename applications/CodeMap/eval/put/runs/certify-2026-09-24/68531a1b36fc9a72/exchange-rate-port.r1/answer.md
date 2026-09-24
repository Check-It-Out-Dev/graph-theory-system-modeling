Added PLN currency conversion behind a swappable port:

- `currency/port/ExchangeRatePort.java` — new port interface, `Optional<BigDecimal> rateToPln(String)`.
- `currency/adapter/FixedExchangeRateProperties.java` — `@ConfigurationProperties(prefix="currency.fixed-rates")` holding `Map<String, BigDecimal> rates`.
- `currency/adapter/FixedExchangeRateAdapter.java` — implements the port; normalizes ISO codes to a uppercase map once in the constructor, PLN always 1.
- `currency/adapter/FixedExchangeRateConfig.java` — `@EnableConfigurationProperties` registration, following `FakturowniaConfig`.
- `currency/CurrencyConversionService.java` — `toPln(amount, isoCode)`, HALF_UP to 2 decimals, throws `BusinessRuleTranslatableException` with key `error.business.exchange_rate_unavailable` when no rate exists.
- `messages_en.properties` / `messages_pl.properties` — new key added to both.
- `application.yml` — sample `currency.fixed-rates.rates` (USD/EUR/GBP) config block.
- New tests: `unit/currency/adapter/FixedExchangeRateAdapterUnitTest.java`, `unit/service/CurrencyConversionServiceUnitTest.java`.

Ran: `mvnw test -Ptest -Dtest=FixedExchangeRateAdapterUnitTest,CurrencyConversionServiceUnitTest,CurrencyServiceUnitTest` (pass) and `mvnw test-compile -Ptest` (green). No files committed.