package com.sm.instagram.platform.currency.port;

import java.math.BigDecimal;
import java.util.Optional;

/**
 * Port interface for looking up an exchange rate to PLN.
 * Primary implementation: {@link com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter} (fixed configuration).
 * Swappable by implementing this interface in a new adapter backed by a live rate provider.
 */
public interface ExchangeRatePort {

    /**
     * Looks up the rate that converts one unit of the given currency to PLN.
     *
     * @param isoCode ISO 4217 currency code, matched case-insensitively
     * @return the rate to PLN, or empty when no rate is known for the code
     */
    Optional<BigDecimal> rateToPln(String isoCode);
}
