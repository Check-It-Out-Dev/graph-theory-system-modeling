Clean pass (quiet mode, no failures). The build compiles and all relevant tests pass.

Summary:
- Added `currency/port/ExchangeRatePort.java`, `currency/adapter/FixedExchangeRateProperties.java` (bound to `currency.fixed-rates`), `currency/adapter/FixedExchangeRateAdapter.java` (case-insensitive lookup, PLN=1), and `currency/CurrencyConversionService.java` (HALF_UP rounding to 2 decimals, throws `BusinessRuleTranslatableException` with key `error.business.exchange_rate_unavailable`).
- Added the new message key to `messages_en.properties` and `messages_pl.properties`.
- Added a `currency.fixed-rates.rates` block (USD/EUR/GBP) to `application.yml`.
- Added `src/test/java/.../unit/service/CurrencyConversionServiceUnitTest.java` (MockitoExtension, no Spring context) covering conversion/rounding, the missing-rate exception, and the adapter's case-insensitivity/PLN default.
- Ran `CurrencyConversionServiceUnitTest` (3/3 passed) and existing `CurrencyServiceUnitTest`/`AddressAndCurrencyUnitTest` (pass, no regressions).