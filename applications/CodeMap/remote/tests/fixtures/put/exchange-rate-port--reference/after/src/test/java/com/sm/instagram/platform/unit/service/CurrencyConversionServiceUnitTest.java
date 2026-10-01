package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.exceptions.TranslatableException;
import com.sm.instagram.platform.currency.CurrencyConversionService;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateProperties;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class CurrencyConversionServiceUnitTest {

    @Mock
    private ExchangeRatePort exchangeRatePort;

    @InjectMocks
    private CurrencyConversionService service;

    @Test
    void convertsWithTheRateAndRoundsHalfUp() {
        when(exchangeRatePort.rateToPln("EUR")).thenReturn(Optional.of(new BigDecimal("4.3055")));
        assertThat(service.toPln(new BigDecimal("10"), "EUR")).isEqualByComparingTo("43.06");
    }

    @Test
    void missingRateIsATranslatableError() {
        when(exchangeRatePort.rateToPln("XYZ")).thenReturn(Optional.empty());
        assertThatThrownBy(() -> service.toPln(BigDecimal.ONE, "XYZ"))
                .isInstanceOf(TranslatableException.class)
                .extracting(e -> ((TranslatableException) e).getMessageKey())
                .isEqualTo("error.business.exchange_rate_unavailable");
    }

    @Test
    void fixedAdapterMatchesCaseInsensitivelyAndKnowsPln() {
        FixedExchangeRateProperties properties = new FixedExchangeRateProperties();
        properties.setRates(Map.of("EUR", new BigDecimal("4.30")));
        FixedExchangeRateAdapter adapter = new FixedExchangeRateAdapter(properties);

        assertThat(adapter.rateToPln("eur")).contains(new BigDecimal("4.30"));
        assertThat(adapter.rateToPln("PLN")).contains(BigDecimal.ONE);
        assertThat(adapter.rateToPln("USD")).isEmpty();
    }
}
