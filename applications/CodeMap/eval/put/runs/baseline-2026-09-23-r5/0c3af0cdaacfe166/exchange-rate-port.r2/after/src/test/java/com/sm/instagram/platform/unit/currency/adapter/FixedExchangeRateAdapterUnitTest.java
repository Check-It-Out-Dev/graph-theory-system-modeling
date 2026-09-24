package com.sm.instagram.platform.unit.currency.adapter;

import com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateProperties;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
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

    private FixedExchangeRateAdapter adapter;

    @BeforeEach
    void setUp() {
        FixedExchangeRateProperties properties = new FixedExchangeRateProperties();
        properties.setRates(Map.of("USD", new BigDecimal("3.95")));
        adapter = new FixedExchangeRateAdapter(properties);
    }

    @Test
    @DisplayName("returns the configured rate")
    void returnsConfiguredRate() {
        assertThat(adapter.rateToPln("USD")).contains(new BigDecimal("3.95"));
    }

    @Test
    @DisplayName("matches the ISO code case-insensitively")
    void matchesCaseInsensitively() {
        assertThat(adapter.rateToPln("usd")).contains(new BigDecimal("3.95"));
    }

    @Test
    @DisplayName("PLN is always 1, even if not configured")
    void plnIsAlwaysOne() {
        assertThat(adapter.rateToPln("pln")).contains(BigDecimal.ONE);
    }

    @Test
    @DisplayName("an unconfigured code returns empty")
    void unknownCodeReturnsEmpty() {
        assertThat(adapter.rateToPln("XYZ")).isEqualTo(Optional.empty());
    }
}
