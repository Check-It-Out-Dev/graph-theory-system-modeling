package com.sm.instagram.platform.currency.adapter;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

@Getter
@Setter
@Configuration
@ConfigurationProperties(prefix = "currency.fixed-rates")
public class FixedExchangeRateProperties {

    private Map<String, BigDecimal> rates = new HashMap<>();
}
