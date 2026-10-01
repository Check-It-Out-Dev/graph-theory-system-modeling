Added a swappable exchange-rate port for PLN conversion, backed by a fixed-configuration adapter.

- `currency/port/ExchangeRatePort.java` — new port, `Optional<BigDecimal> rateToPln(String isoCode)`.
- `currency/adapter/FixedExchangeRateProperties.java` — `@ConfigurationProperties(prefix = "currency.fixed-rates")`, normalizes rate keys to upper case once so the adapter does a direct map lookup.
- `currency/adapter/FixedExchangeRateAdapter.java` — implements the port; case-insensitive lookup, `PLN` always `1`.
- `currency/adapter/FixedExchangeRateConfig.java` — registers the properties class (Fakturownia-style).
- `currency/CurrencyConversionService.java` — `toPln(amount, isoCode)`: multiplies by rate, HALF_UP to 2 decimals; throws `BusinessRuleTranslatableException` with key `error.business.exchange_rate_unavailable` when no rate exists.
- Added the new message key to `messages_en.properties` and `messages_pl.properties`, and a `currency.fixed-rates` sample block to `application.yml`.
- New tests: `CurrencyConversionServiceUnitTest`, `adapter/FixedExchangeRateAdapterUnitTest`, `adapter/FixedExchangeRatePropertiesUnitTest` — all pass (`mvnw test -Ptest`); `mvnw test-compile -Ptest` is green.