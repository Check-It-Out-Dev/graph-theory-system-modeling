package com.sm.instagram.platform.unit.currency;

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
        properties.setRates(Map.of("USD", new BigDecimal("4.05"), "EUR", new BigDecimal("4.35")));
        adapter = new FixedExchangeRateAdapter(properties);
    }

    @Test
    @DisplayName("PLN always resolves to 1 regardless of configuration")
    void plnAlwaysResolvesToOne() {
        assertThat(adapter.rateToPln("PLN")).contains(BigDecimal.ONE);
        assertThat(adapter.rateToPln("pln")).contains(BigDecimal.ONE);
    }

    @Test
    @DisplayName("matches a configured ISO code case-insensitively")
    void matchesConfiguredCodeCaseInsensitively() {
        assertThat(adapter.rateToPln("usd")).contains(new BigDecimal("4.05"));
        assertThat(adapter.rateToPln("USD")).contains(new BigDecimal("4.05"));
    }

    @Test
    @DisplayName("returns empty when no rate is configured for the code")
    void returnsEmptyWhenRateMissing() {
        assertThat(adapter.rateToPln("GBP")).isEqualTo(Optional.empty());
    }
}
