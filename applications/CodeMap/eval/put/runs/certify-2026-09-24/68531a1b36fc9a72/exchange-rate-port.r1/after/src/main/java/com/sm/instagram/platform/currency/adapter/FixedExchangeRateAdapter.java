package com.sm.instagram.platform.currency.adapter;

import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Fixed-rate adapter implementing {@link ExchangeRatePort}.
 * Rates come from configuration for now; a future adapter can call an external
 * provider without any change to the code that converts.
 */
@Component
public class FixedExchangeRateAdapter implements ExchangeRatePort {

    private static final String PLN = "PLN";

    private final Map<String, BigDecimal> ratesByIsoCode;

    public FixedExchangeRateAdapter(FixedExchangeRateProperties properties) {
        this.ratesByIsoCode = new HashMap<>();
        properties.getRates().forEach((isoCode, rate) ->
                ratesByIsoCode.put(isoCode.toUpperCase(Locale.ROOT), rate));
        this.ratesByIsoCode.put(PLN, BigDecimal.ONE);
    }

    @Override
    public Optional<BigDecimal> rateToPln(String isoCode) {
        if (isoCode == null) {
            return Optional.empty();
        }
        return Optional.ofNullable(ratesByIsoCode.get(isoCode.toUpperCase(Locale.ROOT)));
    }
}
