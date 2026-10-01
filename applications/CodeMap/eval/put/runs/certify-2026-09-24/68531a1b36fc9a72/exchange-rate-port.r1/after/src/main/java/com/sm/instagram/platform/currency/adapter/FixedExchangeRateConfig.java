package com.sm.instagram.platform.currency.adapter;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableConfigurationProperties(FixedExchangeRateProperties.class)
public class FixedExchangeRateConfig {
}
