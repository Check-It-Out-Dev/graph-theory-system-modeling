package com.sm.instagram.platform.unit.currency.adapter;

import com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateProperties;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for FixedExchangeRateAdapter.
 * Tests case-insensitive lookup, the fixed PLN rate and unknown currencies.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("FixedExchangeRateAdapter")
class FixedExchangeRateAdapterUnitTest {

    @Nested
    @DisplayName("Known currency")
    class KnownCurrency {

        @Test
        @DisplayName("should return the configured rate for an exact ISO code")
        void shouldReturnConfiguredRateForExactIsoCode() {
            FixedExchangeRateAdapter adapter = adapterWithRates(Map.of("USD", new BigDecimal("4.05")));

            Optional<BigDecimal> result = adapter.rateToPln("USD");

            assertThat(result).contains(new BigDecimal("4.05"));
        }

        @Test
        @DisplayName("should match the ISO code case-insensitively")
        void shouldMatchIsoCodeCaseInsensitively() {
            FixedExchangeRateAdapter adapter = adapterWithRates(Map.of("eur", new BigDecimal("4.35")));

            Optional<BigDecimal> result = adapter.rateToPln("Eur");

            assertThat(result).contains(new BigDecimal("4.35"));
        }
    }

    @Nested
    @DisplayName("PLN")
    class Pln {

        @Test
        @DisplayName("should always be rate 1, even when not configured")
        void shouldAlwaysBeRateOne() {
            FixedExchangeRateAdapter adapter = adapterWithRates(Map.of("USD", new BigDecimal("4.05")));

            Optional<BigDecimal> result = adapter.rateToPln("pln");

            assertThat(result).contains(BigDecimal.ONE);
        }
    }

    @Nested
    @DisplayName("Unknown currency")
    class UnknownCurrency {

        @Test
        @DisplayName("should return empty when no rate is configured")
        void shouldReturnEmptyWhenNoRateConfigured() {
            FixedExchangeRateAdapter adapter = adapterWithRates(Map.of("USD", new BigDecimal("4.05")));

            Optional<BigDecimal> result = adapter.rateToPln("JPY");

            assertThat(result).isEmpty();
        }
    }

    private FixedExchangeRateAdapter adapterWithRates(Map<String, BigDecimal> rates) {
        FixedExchangeRateProperties properties = new FixedExchangeRateProperties();
        properties.setRates(new LinkedHashMap<>(rates));
        return new FixedExchangeRateAdapter(properties);
    }
}
