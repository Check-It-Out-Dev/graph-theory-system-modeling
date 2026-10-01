package com.sm.instagram.platform.currency.port;

import java.math.BigDecimal;
import java.util.Optional;

/**
 * Port for looking up currency exchange rates to PLN.
 * Implementations may come from fixed configuration or an external provider.
 */
public interface ExchangeRatePort {

    /**
     * Looks up the exchange rate to PLN for the given ISO currency code.
     *
     * @param isoCode ISO 4217 currency code, matched case-insensitively
     * @return the rate to PLN, or empty if no rate is known for the code
     */
    Optional<BigDecimal> rateToPln(String isoCode);
}
