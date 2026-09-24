package com.sm.instagram.platform.currency.port;

import java.math.BigDecimal;
import java.util.Optional;

/**
 * Port for exchange-rate lookups.
 * Primary implementation: fixed rates in configuration.
 * Abstracted to allow swapping the rate provider without changing business logic.
 */
public interface ExchangeRatePort {

    /**
     * Looks up the rate that converts one unit of the given currency to PLN.
     *
     * @param isoCode ISO 4217 currency code, matched case-insensitively
     * @return the rate, or empty when the currency is not known to the provider
     */
    Optional<BigDecimal> rateToPln(String isoCode);
}
