package com.sm.instagram.platform.currency.adapter;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

/**
 * Fixed exchange rates to PLN, configured until an external rate provider replaces them.
 * Keys are ISO 4217 currency codes, values are the rate to PLN.
 */
@Data
@Configuration
@ConfigurationProperties(prefix = "currency.fixed-rates")
public class FixedExchangeRateProperties {

    private Map<String, BigDecimal> rates = new HashMap<>();
}
