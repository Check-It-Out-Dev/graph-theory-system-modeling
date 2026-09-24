package com.sm.instagram.platform.unit.currency;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.currency.CurrencyConversionService;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for CurrencyConversionService.
 * Tests rounding and the error thrown when no rate is available.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("CurrencyConversionService")
class CurrencyConversionServiceUnitTest {

    @Mock
    private ExchangeRatePort exchangeRatePort;

    private CurrencyConversionService service;

    @Nested
    @DisplayName("Known rate")
    class KnownRate {

        @Test
        @DisplayName("should multiply the amount by the rate and round half up to 2 decimals")
        void shouldConvertAndRoundHalfUp() {
            service = new CurrencyConversionService(exchangeRatePort);
            when(exchangeRatePort.rateToPln("USD")).thenReturn(Optional.of(new BigDecimal("4.055")));

            BigDecimal result = service.toPln(new BigDecimal("10.00"), "USD");

            assertThat(result).isEqualByComparingTo(new BigDecimal("40.55"));
        }

        @Test
        @DisplayName("should round 0.005 up on the boundary")
        void shouldRoundHalfUpOnBoundary() {
            service = new CurrencyConversionService(exchangeRatePort);
            when(exchangeRatePort.rateToPln("EUR")).thenReturn(Optional.of(new BigDecimal("1.005")));

            BigDecimal result = service.toPln(BigDecimal.ONE, "EUR");

            assertThat(result).isEqualByComparingTo(new BigDecimal("1.01"));
        }
    }

    @Nested
    @DisplayName("Unknown rate")
    class UnknownRate {

        @Test
        @DisplayName("should throw a translatable business exception carrying the ISO code")
        void shouldThrowWhenRateUnavailable() {
            service = new CurrencyConversionService(exchangeRatePort);
            when(exchangeRatePort.rateToPln("JPY")).thenReturn(Optional.empty());

            BusinessRuleTranslatableException exception = org.junit.jupiter.api.Assertions.assertThrows(
                    BusinessRuleTranslatableException.class,
                    () -> service.toPln(new BigDecimal("10.00"), "JPY"));

            assertThat(exception.getMessageKey()).isEqualTo("error.business.exchange_rate_unavailable");
            assertThat(exception.getArgs()).containsExactly("JPY");
            verify(exchangeRatePort).rateToPln("JPY");
        }
    }
}
