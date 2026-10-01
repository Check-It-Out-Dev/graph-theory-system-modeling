package com.sm.instagram.platform.currency.adapter;

import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Exchange rate adapter backed by fixed rates from configuration.
 * PLN always maps to 1, regardless of what is configured.
 */
@Component
public class FixedExchangeRateAdapter implements ExchangeRatePort {

    private final FixedExchangeRateProperties properties;

    public FixedExchangeRateAdapter(FixedExchangeRateProperties properties) {
        this.properties = properties;
    }

    @Override
    public Optional<BigDecimal> rateToPln(String isoCode) {
        if (isoCode == null) {
            return Optional.empty();
        }

        String normalized = isoCode.toUpperCase(Locale.ROOT);
        if ("PLN".equals(normalized)) {
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
