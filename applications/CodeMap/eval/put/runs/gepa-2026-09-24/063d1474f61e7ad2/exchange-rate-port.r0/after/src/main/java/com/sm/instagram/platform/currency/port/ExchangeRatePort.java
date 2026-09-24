package com.sm.instagram.platform.currency.port;

import java.math.BigDecimal;
import java.util.Optional;

/**
 * Port for retrieving currency exchange rates to PLN.
 * Implementations may source rates from configuration or an external provider.
 */
public interface ExchangeRatePort {

    /**
     * Looks up the exchange rate to PLN for a given ISO currency code.
     *
     * @param isoCode the ISO 4217 currency code
     * @return the rate to PLN, or empty if no rate is available for the code
     */
    Optional<BigDecimal> rateToPln(String isoCode);
}
