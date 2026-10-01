package com.sm.instagram.platform.currency.adapter;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Map;

/**
 * Configuration-fixed exchange rates to PLN, keyed by ISO 4217 currency code.
 */
@Component
@ConfigurationProperties(prefix = "currency.fixed-rates")
@Getter
@Setter
public class FixedExchangeRateProperties {

    private Map<String, BigDecimal> rates = Map.of();
}
