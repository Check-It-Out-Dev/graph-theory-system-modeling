package com.sm.instagram.platform.currency.adapter;

import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * ExchangeRatePort backed by rates fixed in configuration ({@link FixedExchangeRateProperties}).
 */
@Component
@RequiredArgsConstructor
public class FixedExchangeRateAdapter implements ExchangeRatePort {

    private static final String PLN = "PLN";

    private final FixedExchangeRateProperties properties;

    @Override
    public Optional<BigDecimal> rateToPln(String isoCode) {
        if (isoCode == null || isoCode.isBlank()) {
            return Optional.empty();
        }
        String code = isoCode.trim().toUpperCase(Locale.ROOT);
        if (PLN.equals(code)) {
            return Optional.of(BigDecimal.ONE);
        }
        for (Map.Entry<String, BigDecimal> entry : properties.getRates().entrySet()) {
            if (entry.getKey().equalsIgnoreCase(code)) {
                return Optional.ofNullable(entry.getValue());
            }
        }
        return Optional.empty();
    }
}
