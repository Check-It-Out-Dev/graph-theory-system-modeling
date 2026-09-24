package com.sm.instagram.platform.unit.currency.adapter;

import com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateProperties;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(MockitoExtension.class)
@DisplayName("FixedExchangeRateAdapter")
class FixedExchangeRateAdapterUnitTest {

    private FixedExchangeRateProperties properties;
    private FixedExchangeRateAdapter adapter;

    @BeforeEach
    void setUp() {
        properties = new FixedExchangeRateProperties();
        properties.setRates(Map.of("EUR", new BigDecimal("4.30")));
        adapter = new FixedExchangeRateAdapter(properties);
    }

    @Nested
    @DisplayName("Known rate")
    class KnownRate {

        @Test
        @DisplayName("should return the configured rate for a matching code")
        void shouldReturnConfiguredRate() {
            Optional<BigDecimal> result = adapter.rateToPln("EUR");

            assertThat(result).contains(new BigDecimal("4.30"));
        }

        @Test
        @DisplayName("should match the ISO code case-insensitively")
        void shouldMatchCaseInsensitively() {
            Optional<BigDecimal> result = adapter.rateToPln("eur");

            assertThat(result).contains(new BigDecimal("4.30"));
        }
    }

    @Nested
    @DisplayName("PLN")
    class Pln {

        @Test
        @DisplayName("should always return a rate of 1")
        void shouldReturnOneForPln() {
            Optional<BigDecimal> result = adapter.rateToPln("pln");

            assertThat(result).contains(BigDecimal.ONE);
        }
    }

    @Nested
    @DisplayName("Missing rate")
    class MissingRate {

        @Test
        @DisplayName("should return empty when no rate is configured")
        void shouldReturnEmptyForUnknownCode() {
            Optional<BigDecimal> result = adapter.rateToPln("XYZ");

            assertThat(result).isEmpty();
        }
    }
}
