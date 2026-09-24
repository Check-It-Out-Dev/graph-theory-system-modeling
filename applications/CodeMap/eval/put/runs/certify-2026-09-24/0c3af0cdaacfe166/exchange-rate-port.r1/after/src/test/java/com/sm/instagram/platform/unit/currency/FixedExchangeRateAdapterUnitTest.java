package com.sm.instagram.platform.unit.currency;

import com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateProperties;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("FixedExchangeRateAdapter — reads rates from configuration")
class FixedExchangeRateAdapterUnitTest {

    private FixedExchangeRateAdapter adapter;

    @BeforeEach
    void setUp() {
        FixedExchangeRateProperties properties = new FixedExchangeRateProperties();
        properties.setRates(Map.of("USD", new BigDecimal("4.00")));
        adapter = new FixedExchangeRateAdapter(properties);
    }

    @Test
    @DisplayName("PLN always resolves to 1, even if not configured")
    void plnIsAlwaysOne() {
        assertThat(adapter.rateToPln("PLN")).contains(BigDecimal.ONE);
    }

    @Test
    @DisplayName("matches ISO codes case-insensitively")
    void matchesCaseInsensitively() {
        Optional<BigDecimal> rate = adapter.rateToPln("usd");

        assertThat(rate).isPresent();
        assertThat(rate.get()).isEqualByComparingTo("4.00");
    }

    @Test
    @DisplayName("returns empty when the currency has no configured rate")
    void returnsEmptyForUnknownCurrency() {
        assertThat(adapter.rateToPln("XYZ")).isEmpty();
    }
}
