package com.sm.instagram.platform.unit.currency.adapter;

import com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateProperties;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("FixedExchangeRateAdapter")
class FixedExchangeRateAdapterUnitTest {

    private FixedExchangeRateProperties properties;
    private FixedExchangeRateAdapter adapter;

    @BeforeEach
    void setUp() {
        properties = new FixedExchangeRateProperties();
        Map<String, BigDecimal> rates = new HashMap<>();
        rates.put("EUR", new BigDecimal("4.3210"));
        properties.setRates(rates);
        adapter = new FixedExchangeRateAdapter(properties);
    }

    @Test
    @DisplayName("should return the configured rate")
    void shouldReturnConfiguredRate() {
        Optional<BigDecimal> result = adapter.rateToPln("EUR");

        assertThat(result).isPresent();
        assertThat(result.get()).isEqualByComparingTo(new BigDecimal("4.3210"));
    }

    @Test
    @DisplayName("should match the ISO code case-insensitively")
    void shouldMatchCaseInsensitively() {
        Optional<BigDecimal> result = adapter.rateToPln("eur");

        assertThat(result).isPresent();
        assertThat(result.get()).isEqualByComparingTo(new BigDecimal("4.3210"));
    }

    @Test
    @DisplayName("should always return 1 for PLN even when not configured")
    void shouldReturnOneForPln() {
        Optional<BigDecimal> result = adapter.rateToPln("pln");

        assertThat(result).isPresent();
        assertThat(result.get()).isEqualByComparingTo(BigDecimal.ONE);
    }

    @Test
    @DisplayName("should return empty when no rate is configured for the code")
    void shouldReturnEmptyWhenNoRateConfigured() {
        Optional<BigDecimal> result = adapter.rateToPln("XYZ");

        assertThat(result).isEmpty();
    }
}
