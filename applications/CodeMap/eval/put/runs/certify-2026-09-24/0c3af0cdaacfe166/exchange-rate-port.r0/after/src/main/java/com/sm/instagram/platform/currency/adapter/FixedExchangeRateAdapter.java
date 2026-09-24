package com.sm.instagram.platform.currency.adapter;

import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Map;
import java.util.Optional;

/**
 * Exchange rate adapter backed by rates fixed in configuration.
 * Stands in until an external rate provider is wired in.
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
        for (Map.Entry<String, BigDecimal> entry : properties.getRates().entrySet()) {
            if (entry.getKey().equalsIgnoreCase(isoCode)) {
                return Optional.of(entry.getValue());
            }
        }
        return Optional.empty();
    }
}
