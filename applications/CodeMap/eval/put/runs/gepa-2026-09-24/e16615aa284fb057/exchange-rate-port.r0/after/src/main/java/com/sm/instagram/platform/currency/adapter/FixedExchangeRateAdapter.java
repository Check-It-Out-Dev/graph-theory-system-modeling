package com.sm.instagram.platform.currency.adapter;

import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Map;
import java.util.Optional;

/**
 * Reads exchange rates to PLN from fixed configuration values.
 * Non-blocking: rates come from the {@code currency.fixed-rates} properties, no external call.
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
        String normalized = isoCode.toUpperCase();
        if (PLN.equals(normalized)) {
            return Optional.of(BigDecimal.ONE);
        }
        for (Map.Entry<String, BigDecimal> entry : properties.getRates().entrySet()) {
            if (entry.getKey().equalsIgnoreCase(normalized)) {
                return Optional.of(entry.getValue());
            }
        }
        return Optional.empty();
    }
}
