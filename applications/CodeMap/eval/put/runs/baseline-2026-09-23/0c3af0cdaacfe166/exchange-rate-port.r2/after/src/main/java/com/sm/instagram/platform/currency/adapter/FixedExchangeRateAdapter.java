package com.sm.instagram.platform.currency.adapter;

import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Map;
import java.util.Optional;

/**
 * Reads exchange rates to PLN from {@link FixedExchangeRateProperties}. PLN itself is always 1,
 * regardless of what configuration says.
 */
@Component
public class FixedExchangeRateAdapter implements ExchangeRatePort {

    private static final String PLN = "PLN";

    private final FixedExchangeRateProperties properties;

    public FixedExchangeRateAdapter(FixedExchangeRateProperties properties) {
        this.properties = properties;
    }

    @Override
    public Optional<BigDecimal> rateToPln(String isoCode) {
        if (isoCode == null) {
            return Optional.empty();
        }
        if (PLN.equalsIgnoreCase(isoCode)) {
            return Optional.of(BigDecimal.ONE);
        }
        return properties.getRates().entrySet().stream()
                .filter(entry -> entry.getKey().equalsIgnoreCase(isoCode))
                .map(Map.Entry::getValue)
                .findFirst();
    }
}
