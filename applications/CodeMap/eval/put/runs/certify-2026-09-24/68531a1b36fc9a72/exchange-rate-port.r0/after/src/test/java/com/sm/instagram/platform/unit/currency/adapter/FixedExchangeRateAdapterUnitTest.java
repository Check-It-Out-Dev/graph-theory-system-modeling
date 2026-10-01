package com.sm.instagram.platform.unit.currency.adapter;

import com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateProperties;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(MockitoExtension.class)
class FixedExchangeRateAdapterUnitTest {

    private FixedExchangeRateAdapter adapter;

    @BeforeEach
    void setUp() {
        FixedExchangeRateProperties properties = new FixedExchangeRateProperties();
        properties.setRates(Map.of(
                "EUR", new BigDecimal("4.30"),
                "usd", new BigDecimal("3.95")
        ));
        adapter = new FixedExchangeRateAdapter(properties);
    }

    @Test
    void shouldReturnConfiguredRateRegardlessOfCase() {
        assertThat(adapter.rateToPln("eur")).contains(new BigDecimal("4.30"));
        assertThat(adapter.rateToPln("USD")).contains(new BigDecimal("3.95"));
    }

    @Test
    void shouldAlwaysReturnOneForPln() {
        assertThat(adapter.rateToPln("pln")).contains(BigDecimal.ONE);
        assertThat(adapter.rateToPln("PLN")).contains(BigDecimal.ONE);
    }

    @Test
    void shouldOverrideConfiguredPlnRateWithOne() {
        FixedExchangeRateProperties properties = new FixedExchangeRateProperties();
        properties.setRates(Map.of("PLN", new BigDecimal("2.00")));
        FixedExchangeRateAdapter adapterWithMisconfiguredPln = new FixedExchangeRateAdapter(properties);

        assertThat(adapterWithMisconfiguredPln.rateToPln("PLN")).contains(BigDecimal.ONE);
    }

    @Test
    void shouldReturnEmptyForUnknownCurrency() {
        assertThat(adapter.rateToPln("GBP")).isEmpty();
    }

    @Test
    void shouldReturnEmptyForNullIsoCode() {
        assertThat(adapter.rateToPln(null)).isEmpty();
    }
}
