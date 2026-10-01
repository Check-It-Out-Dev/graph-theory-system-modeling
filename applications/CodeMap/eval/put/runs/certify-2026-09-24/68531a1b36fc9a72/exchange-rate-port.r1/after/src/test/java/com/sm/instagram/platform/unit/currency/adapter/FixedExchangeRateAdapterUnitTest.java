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
        properties.setRates(Map.of(
                "USD", new BigDecimal("4.05"),
                "eur", new BigDecimal("4.35")));
        adapter = new FixedExchangeRateAdapter(properties);
    }

    @Test
    @DisplayName("should return the configured rate for an upper case ISO code")
    void shouldReturnRateForUpperCaseCode() {
        Optional<BigDecimal> rate = adapter.rateToPln("USD");

        assertThat(rate).contains(new BigDecimal("4.05"));
    }

    @Test
    @DisplayName("should match ISO code case-insensitively")
    void shouldMatchCaseInsensitively() {
        assertThat(adapter.rateToPln("usd")).contains(new BigDecimal("4.05"));
        assertThat(adapter.rateToPln("EUR")).contains(new BigDecimal("4.35"));
    }

    @Test
    @DisplayName("should always report PLN at rate 1")
    void shouldReportPlnAtRateOne() {
        assertThat(adapter.rateToPln("PLN")).contains(BigDecimal.ONE);
        assertThat(adapter.rateToPln("pln")).contains(BigDecimal.ONE);
    }

    @Test
    @DisplayName("should return empty for a currency with no configured rate")
    void shouldReturnEmptyForUnknownCode() {
        assertThat(adapter.rateToPln("XYZ")).isEmpty();
    }

    @Test
    @DisplayName("should return empty for a null ISO code")
    void shouldReturnEmptyForNullCode() {
        assertThat(adapter.rateToPln(null)).isEmpty();
    }
}
