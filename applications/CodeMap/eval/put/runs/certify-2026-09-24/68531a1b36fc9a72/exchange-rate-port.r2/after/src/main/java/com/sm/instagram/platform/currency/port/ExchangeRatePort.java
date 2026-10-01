package com.sm.instagram.platform.currency.port;

import java.math.BigDecimal;
import java.util.Optional;

/**
 * Port for looking up the exchange rate to convert an amount in a given currency to PLN.
 * Primary implementation: fixed rates from configuration, replaceable by a live provider.
 */
public interface ExchangeRatePort {

    /**
     * Looks up the rate to multiply an amount in the given currency by to get PLN.
     *
     * @param isoCode ISO 4217 currency code, matched case-insensitively
     * @return the rate to PLN, or empty when the currency is not known
     */
    Optional<BigDecimal> rateToPln(String isoCode);
}
