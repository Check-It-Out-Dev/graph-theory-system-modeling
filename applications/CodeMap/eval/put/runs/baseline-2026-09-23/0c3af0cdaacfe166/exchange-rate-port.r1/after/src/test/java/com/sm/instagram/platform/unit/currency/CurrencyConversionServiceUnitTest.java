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
@DisplayName("CurrencyConversionService — converts to PLN via the exchange rate port")
class CurrencyConversionServiceUnitTest {

    @Mock private ExchangeRatePort exchangeRatePort;
    private CurrencyConversionService service;

    @BeforeEach
    void setUp() {
        service = new CurrencyConversionService(exchangeRatePort);
    }

    @Test
    @DisplayName("multiplies the amount by the rate and rounds to 2 decimals HALF_UP")
    void convertsUsingRate() {
        when(exchangeRatePort.rateToPln("USD")).thenReturn(Optional.of(new BigDecimal("4.005")));

        BigDecimal result = service.toPln(new BigDecimal("10"), "USD");

        assertThat(result).isEqualByComparingTo("40.05");
    }

    @Test
    @DisplayName("rounds .5 up, not to even")
    void roundsHalfUp() {
        when(exchangeRatePort.rateToPln("USD")).thenReturn(Optional.of(new BigDecimal("1")));

        BigDecimal result = service.toPln(new BigDecimal("10.005"), "USD");

        assertThat(result).isEqualByComparingTo("10.01");
    }

    @Test
    @DisplayName("throws a translatable business rule exception with the ISO code when no rate exists")
    void throwsWhenRateUnavailable() {
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
