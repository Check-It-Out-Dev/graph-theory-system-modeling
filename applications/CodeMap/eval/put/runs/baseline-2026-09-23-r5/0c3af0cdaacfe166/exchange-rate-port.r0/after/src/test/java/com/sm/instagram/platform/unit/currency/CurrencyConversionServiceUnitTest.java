package com.sm.instagram.platform.unit.currency;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.currency.CurrencyConversionService;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@DisplayName("CurrencyConversionService")
class CurrencyConversionServiceUnitTest {

    @Mock private ExchangeRatePort exchangeRatePort;
    private CurrencyConversionService service;

    @BeforeEach
    void setUp() {
        service = new CurrencyConversionService(exchangeRatePort);
    }

    @Test
    @DisplayName("converts an amount by multiplying by the rate and rounds to 2 decimals HALF_UP")
    void convertsAmountAndRoundsHalfUp() {
        when(exchangeRatePort.rateToPln("USD")).thenReturn(Optional.of(new BigDecimal("4.055")));

        BigDecimal result = service.toPln(new BigDecimal("10"), "USD");

        assertThat(result).isEqualByComparingTo("40.55");
    }

    @Test
    @DisplayName("rounds 0.005 up on the half-up boundary")
    void roundsHalfUpBoundary() {
        when(exchangeRatePort.rateToPln("USD")).thenReturn(Optional.of(new BigDecimal("1")));

        BigDecimal result = service.toPln(new BigDecimal("1.005"), "USD");

        assertThat(result).isEqualByComparingTo("1.01");
    }

    @Test
    @DisplayName("throws a translatable exception with the ISO code when no rate is available")
    void throwsWhenNoRateAvailable() {
        when(exchangeRatePort.rateToPln("XYZ")).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.toPln(BigDecimal.TEN, "XYZ"))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> {
                    BusinessRuleTranslatableException translatable = (BusinessRuleTranslatableException) ex;
                    assertThat(translatable.getMessageKey()).isEqualTo("error.business.exchange_rate_unavailable");
                    assertThat(translatable.getArgs()).containsExactly("XYZ");
                });
    }
}
