package com.sm.instagram.platform.currency.port;

import java.math.BigDecimal;
import java.util.Optional;

/**
 * Port interface for looking up the exchange rate to PLN for a currency.
 * Primary implementation: {@link com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter} (configuration-fixed rates).
 * Swappable by implementing this interface in a new adapter and using {@code @Primary} or {@code @Profile}.
 */
public interface ExchangeRatePort {

    /**
     * Looks up the exchange rate to PLN for the given ISO currency code.
     *
     * @param isoCode ISO 4217 currency code, matched case-insensitively
     * @return the rate to PLN, or empty when no rate is available for the code
     */
    Optional<BigDecimal> rateToPln(String isoCode);
}
