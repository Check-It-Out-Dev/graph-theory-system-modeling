package com.sm.instagram.platform.currency.adapter;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(MockitoExtension.class)
class FixedExchangeRateAdapterUnitTest {

    private FixedExchangeRateProperties properties;
    private FixedExchangeRateAdapter adapter;

    @BeforeEach
    void setUp() {
        properties = new FixedExchangeRateProperties();
        properties.setRates(Map.of("EUR", new BigDecimal("4.30"), "USD", new BigDecimal("4.00")));
        adapter = new FixedExchangeRateAdapter(properties);
    }

    @Test
    void shouldReturnConfiguredRateForKnownCode() {
        Optional<BigDecimal> rate = adapter.rateToPln("EUR");

        assertThat(rate).hasValueSatisfying(value -> assertThat(value).isEqualByComparingTo("4.30"));
    }

    @Test
    void shouldMatchIsoCodeCaseInsensitively() {
        Optional<BigDecimal> rate = adapter.rateToPln("eur");

        assertThat(rate).hasValueSatisfying(value -> assertThat(value).isEqualByComparingTo("4.30"));
    }

    @Test
    void shouldAlwaysReturnOneForPlnRegardlessOfConfiguration() {
        Optional<BigDecimal> rate = adapter.rateToPln("pln");

        assertThat(rate).hasValueSatisfying(value -> assertThat(value).isEqualByComparingTo("1"));
    }

    @Test
    void shouldReturnEmptyForUnknownCode() {
        Optional<BigDecimal> rate = adapter.rateToPln("XYZ");

        assertThat(rate).isEmpty();
    }

    @Test
    void shouldReturnEmptyForNullIsoCode() {
        Optional<BigDecimal> rate = adapter.rateToPln(null);

        assertThat(rate).isEmpty();
    }
}
