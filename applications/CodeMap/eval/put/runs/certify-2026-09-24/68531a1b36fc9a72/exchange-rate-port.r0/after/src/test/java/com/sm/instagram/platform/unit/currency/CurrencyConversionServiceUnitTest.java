package com.sm.instagram.platform.unit.currency;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.currency.CurrencyConversionService;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class CurrencyConversionServiceUnitTest {

    @Mock
    private ExchangeRatePort exchangeRatePort;

    private CurrencyConversionService service;

    @BeforeEach
    void setUp() {
        service = new CurrencyConversionService(exchangeRatePort);
    }

    @Test
    void shouldConvertAmountUsingTheProvidedRate() {
        when(exchangeRatePort.rateToPln("EUR")).thenReturn(Optional.of(new BigDecimal("4.30")));

        BigDecimal result = service.toPln(new BigDecimal("100"), "EUR");

        assertThat(result).isEqualByComparingTo("430.00");
    }

    @Test
    void shouldRoundHalfUpToTwoDecimals() {
        when(exchangeRatePort.rateToPln("USD")).thenReturn(Optional.of(BigDecimal.ONE));

        BigDecimal result = service.toPln(new BigDecimal("10.005"), "USD");

        assertThat(result).isEqualByComparingTo("10.01");
    }

    @Test
    void shouldThrowTranslatableExceptionWithIsoCodeWhenRateMissing() {
        when(exchangeRatePort.rateToPln("XYZ")).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.toPln(new BigDecimal("50"), "XYZ"))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> {
                    BusinessRuleTranslatableException translatable = (BusinessRuleTranslatableException) ex;
                    assertThat(translatable.getMessageKey()).isEqualTo("error.business.exchange_rate_unavailable");
                    assertThat(translatable.getArgs()).containsExactly("XYZ");
                });

        verify(exchangeRatePort).rateToPln("XYZ");
    }
}
