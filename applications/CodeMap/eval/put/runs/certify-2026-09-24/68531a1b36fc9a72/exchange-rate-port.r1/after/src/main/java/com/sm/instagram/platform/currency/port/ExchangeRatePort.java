package com.sm.instagram.platform.currency.port;

import java.math.BigDecimal;
import java.util.Optional;

/**
 * Port interface for currency exchange rate lookup.
 * Abstracted to allow swapping the rate provider without changing the code that converts.
 */
public interface ExchangeRatePort {

    /**
     * Looks up the rate to convert one unit of the given currency into PLN.
     *
     * @param isoCode ISO 4217 currency code (e.g. "USD")
     * @return the rate to PLN, or empty when no rate is known for the code
     */
    Optional<BigDecimal> rateToPln(String isoCode);
}
