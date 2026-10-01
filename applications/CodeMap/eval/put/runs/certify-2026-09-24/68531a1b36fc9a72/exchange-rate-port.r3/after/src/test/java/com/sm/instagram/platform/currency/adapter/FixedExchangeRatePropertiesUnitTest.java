package com.sm.instagram.platform.currency.adapter;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(MockitoExtension.class)
class FixedExchangeRatePropertiesUnitTest {

    @Test
    void shouldNormalizeConfiguredKeysToUpperCase() {
        FixedExchangeRateProperties properties = new FixedExchangeRateProperties();

        properties.setRates(Map.of("eur", new BigDecimal("4.30"), "Usd", new BigDecimal("4.00")));

        assertThat(properties.getRates())
                .containsEntry("EUR", new BigDecimal("4.30"))
                .containsEntry("USD", new BigDecimal("4.00"))
                .doesNotContainKey("eur")
                .doesNotContainKey("Usd");
    }

    @Test
    void shouldReplaceRatesWithEmptyMapWhenSetToNull() {
        FixedExchangeRateProperties properties = new FixedExchangeRateProperties();
        properties.setRates(Map.of("EUR", new BigDecimal("4.30")));

        properties.setRates(null);

        assertThat(properties.getRates()).isEmpty();
    }
}
