package com.sm.instagram.platform.unit.currency;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.currency.CurrencyConversionService;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("CurrencyConversionService")
class CurrencyConversionServiceUnitTest {

    @Mock
    private ExchangeRatePort exchangeRatePort;

    @InjectMocks
    private CurrencyConversionService service;

    @Test
    @DisplayName("should convert amount using the rate from the port")
    void shouldConvertAmountUsingRate() {
        when(exchangeRatePort.rateToPln("EUR")).thenReturn(Optional.of(new BigDecimal("4.3210")));

        BigDecimal result = service.toPln(new BigDecimal("10.00"), "EUR");

        assertThat(result).isEqualByComparingTo(new BigDecimal("43.21"));
        verify(exchangeRatePort).rateToPln("EUR");
    }

    @Test
    @DisplayName("should round to 2 decimals using HALF_UP")
    void shouldRoundHalfUp() {
        when(exchangeRatePort.rateToPln("USD")).thenReturn(Optional.of(new BigDecimal("3.995")));

        BigDecimal result = service.toPln(new BigDecimal("1.00"), "USD");

        assertThat(result).isEqualByComparingTo(new BigDecimal("1.00").multiply(new BigDecimal("3.995"))
                .setScale(2, RoundingMode.HALF_UP));
        assertThat(result).isEqualByComparingTo(new BigDecimal("4.00"));
    }

    @Test
    @DisplayName("should throw a translatable exception with the ISO code when no rate exists")
    void shouldThrowWhenNoRateExists() {
        when(exchangeRatePort.rateToPln("XYZ")).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.toPln(new BigDecimal("10.00"), "XYZ"))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> {
                    BusinessRuleTranslatableException translatable = (BusinessRuleTranslatableException) ex;
                    assertThat(translatable.getMessageKey()).isEqualTo("error.business.exchange_rate_unavailable");
                    assertThat(translatable.getArgs()).containsExactly("XYZ");
                });
    }
}
