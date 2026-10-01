package com.sm.instagram.platform.currency.adapter;

import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Locale;
import java.util.Optional;

/**
 * Reads exchange rates to PLN from fixed configuration ({@link FixedExchangeRateProperties}).
 * Implements {@link ExchangeRatePort}; swap in a live-rate adapter behind the same port
 * once a provider is available, without changing {@link com.sm.instagram.platform.currency.CurrencyConversionService}.
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
        String normalized = isoCode.toUpperCase(Locale.ROOT);
        if (PLN.equals(normalized)) {
            return Optional.of(BigDecimal.ONE);
        }
        return Optional.ofNullable(properties.getRates().get(normalized));
    }
}
