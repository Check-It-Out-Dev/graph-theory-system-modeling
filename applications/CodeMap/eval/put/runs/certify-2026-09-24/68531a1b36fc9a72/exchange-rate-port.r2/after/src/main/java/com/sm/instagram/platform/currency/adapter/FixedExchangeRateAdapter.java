package com.sm.instagram.platform.currency.adapter;

import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Fixed-rate exchange rate adapter. Rates are configured statically and can be
 * replaced by a live provider adapter behind {@link ExchangeRatePort} without
 * touching the code that converts amounts.
 */
@Component
public class FixedExchangeRateAdapter implements ExchangeRatePort {

    private static final String PLN = "PLN";

    private final Map<String, BigDecimal> ratesByIsoCode;

    public FixedExchangeRateAdapter(FixedExchangeRateProperties properties) {
        this.ratesByIsoCode = properties.getRates().entrySet().stream()
                .collect(Collectors.toMap(
                        entry -> entry.getKey().toUpperCase(Locale.ROOT),
                        Map.Entry::getValue));
    }

    @Override
    public Optional<BigDecimal> rateToPln(String isoCode) {
        if (isoCode == null) {
            return Optional.empty();
        }

        String normalizedIsoCode = isoCode.toUpperCase(Locale.ROOT);
        if (PLN.equals(normalizedIsoCode)) {
            return Optional.of(BigDecimal.ONE);
        }

        return Optional.ofNullable(ratesByIsoCode.get(normalizedIsoCode));
    }
}
