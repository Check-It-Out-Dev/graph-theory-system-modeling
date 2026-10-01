package com.sm.instagram.platform.currency.adapter;

import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Exchange rate adapter backed by fixed rates in configuration.
 * Rates come from {@code currency.fixed-rates} and will later be replaced
 * by an adapter that fetches rates from an external provider.
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

        String normalizedCode = isoCode.toUpperCase(Locale.ROOT);
        if (PLN.equals(normalizedCode)) {
            return Optional.of(BigDecimal.ONE);
        }

        return properties.getRates().entrySet().stream()
                .filter(entry -> entry.getKey().equalsIgnoreCase(normalizedCode))
                .map(Map.Entry::getValue)
                .findFirst();
    }
}
