package com.sm.instagram.platform.currency.adapter;

import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

@Component
public class FixedExchangeRateAdapter implements ExchangeRatePort {

    private static final String PLN = "PLN";

    private final Map<String, BigDecimal> ratesByIsoCode;

    public FixedExchangeRateAdapter(FixedExchangeRateProperties properties) {
        Map<String, BigDecimal> normalized = new HashMap<>();
        properties.getRates().forEach((isoCode, rate) -> normalized.put(isoCode.toUpperCase(Locale.ROOT), rate));
        normalized.put(PLN, BigDecimal.ONE);
        this.ratesByIsoCode = normalized;
    }

    @Override
    public Optional<BigDecimal> rateToPln(String isoCode) {
        if (isoCode == null) {
            return Optional.empty();
        }
        return Optional.ofNullable(ratesByIsoCode.get(isoCode.toUpperCase(Locale.ROOT)));
    }
}
