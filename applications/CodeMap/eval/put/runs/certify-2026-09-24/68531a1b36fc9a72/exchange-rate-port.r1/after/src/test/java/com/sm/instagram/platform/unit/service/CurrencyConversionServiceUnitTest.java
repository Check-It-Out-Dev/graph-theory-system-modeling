package com.sm.instagram.platform.unit.service;

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

    @Mock
    private ExchangeRatePort exchangeRatePort;

    private CurrencyConversionService service;

    @BeforeEach
    void setUp() {
        service = new CurrencyConversionService(exchangeRatePort);
    }

    @Test
    @DisplayName("should multiply amount by rate and round to 2 decimals HALF_UP")
    void shouldConvertAndRoundHalfUp() {
        when(exchangeRatePort.rateToPln("USD")).thenReturn(Optional.of(new BigDecimal("4.015")));

        BigDecimal result = service.toPln(new BigDecimal("10.00"), "USD");

        // 10.00 * 4.015 = 40.150 -> HALF_UP to 40.15
        assertThat(result).isEqualByComparingTo(new BigDecimal("40.15"));
    }

    @Test
    @DisplayName("should round up on a half-way value")
    void shouldRoundHalfUpOnHalfway() {
        when(exchangeRatePort.rateToPln("EUR")).thenReturn(Optional.of(new BigDecimal("1.125")));

        BigDecimal result = service.toPln(new BigDecimal("1"), "EUR");

        assertThat(result).isEqualByComparingTo(new BigDecimal("1.13"));
    }

    @Test
    @DisplayName("should throw a translatable business rule exception with the ISO code when no rate exists")
    void shouldThrowWhenRateMissing() {
        when(exchangeRatePort.rateToPln("XYZ")).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.toPln(new BigDecimal("5"), "XYZ"))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> {
                    BusinessRuleTranslatableException translatable = (BusinessRuleTranslatableException) ex;
                    assertThat(translatable.getMessageKey()).isEqualTo("error.business.exchange_rate_unavailable");
                    assertThat(translatable.getArgs()).containsExactly("XYZ");
                });
    }
}
