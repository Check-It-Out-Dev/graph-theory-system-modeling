package com.sm.instagram.platform.currency.port;

import java.math.BigDecimal;
import java.util.Optional;

/**
 * Port for looking up the exchange rate to PLN of a currency, identified by its ISO code.
 * Implementations may source rates from configuration or from an external provider.
 */
public interface ExchangeRatePort {

    /**
     * Looks up the exchange rate to PLN for the given ISO currency code.
     *
     * @param isoCode the ISO 4217 currency code (e.g. "USD")
     * @return the rate to PLN, or empty if no rate is known for the code
     */
    Optional<BigDecimal> rateToPln(String isoCode);
}
