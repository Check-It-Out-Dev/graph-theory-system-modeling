package com.sm.instagram.platform.currency.port;

import java.math.BigDecimal;
import java.util.Optional;

/**
 * Port for exchange rates to PLN.
 * Current implementation: FixedExchangeRateAdapter (rates from configuration); an external rate
 * provider replaces it without touching the code that converts.
 */
public interface ExchangeRatePort {

    /**
     * @param isoCode ISO 4217 code of the source currency
     * @return how many PLN one unit of the currency is worth, or empty when no rate is known
     */
    Optional<BigDecimal> rateToPln(String isoCode);
}
