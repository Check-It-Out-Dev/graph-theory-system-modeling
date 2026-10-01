package com.sm.instagram.platform.currency.adapter;

import lombok.Getter;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.math.BigDecimal;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

@Getter
@ConfigurationProperties(prefix = "currency.fixed-rates")
public class FixedExchangeRateProperties {

    private Map<String, BigDecimal> rates = new LinkedHashMap<>();

    public void setRates(Map<String, BigDecimal> rates) {
        Map<String, BigDecimal> normalized = new LinkedHashMap<>();
        if (rates != null) {
            rates.forEach((isoCode, rate) -> normalized.put(isoCode.toUpperCase(Locale.ROOT), rate));
        }
        this.rates = normalized;
    }
}
