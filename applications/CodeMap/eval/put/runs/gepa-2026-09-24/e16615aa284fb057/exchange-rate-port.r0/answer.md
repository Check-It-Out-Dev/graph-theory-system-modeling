No lenient/LENIENT matches, and only the intended files changed. All checks pass.

## Summary

- Added `currency/port/ExchangeRatePort.java`: port interface `Optional<BigDecimal> rateToPln(String isoCode)`.
- Added `currency/adapter/FixedExchangeRateProperties.java`: `@ConfigurationProperties(prefix = "currency.fixed-rates")`, `Map<String, BigDecimal> rates`.
- Added `currency/adapter/FixedExchangeRateAdapter.java`: implements the port, case-insensitive lookup, `PLN` always `1`.
- Added `currency/CurrencyConversionService.java`: `toPln(amount, isoCode)` multiplies by the rate, rounds HALF_UP to 2 decimals, throws `BusinessRuleTranslatableException("error.business.exchange_rate_unavailable", isoCode)` when no rate exists.
- Added key `error.business.exchange_rate_unavailable` to `messages_en.properties` and `messages_pl.properties`.
- Added `unit/currency/CurrencyConversionServiceUnitTest.java` and `unit/currency/adapter/FixedExchangeRateAdapterUnitTest.java` covering conversion, HALF_UP rounding, missing-rate error, case-insensitivity, and the fixed PLN=1 rule.

Tests run: `-Dtest=CurrencyConversionServiceUnitTest,FixedExchangeRateAdapterUnitTest` (pass) and `test-compile -Ptest` (green).