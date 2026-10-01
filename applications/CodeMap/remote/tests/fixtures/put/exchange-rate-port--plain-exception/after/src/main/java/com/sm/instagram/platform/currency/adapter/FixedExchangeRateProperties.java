package com.sm.instagram.platform.currency.adapter;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

/**
 * Fixed exchange rates to PLN, e.g. {@code currency.fixed-rates.rates.EUR=4.30}.
 */
@Getter
@Setter
@Component
@ConfigurationProperties(prefix = "currency.fixed-rates")
public class FixedExchangeRateProperties {

    private Map<String, BigDecimal> rates = new HashMap<>();
}
