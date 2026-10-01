package com.sm.instagram.platform.unit.currency;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.currency.CurrencyConversionService;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
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

    @Nested
    @DisplayName("Main path")
    class MainPath {

        @Test
        @DisplayName("should convert amount by multiplying with the rate")
        void shouldConvertAmountUsingRate() {
            when(exchangeRatePort.rateToPln("EUR")).thenReturn(Optional.of(new BigDecimal("4.30")));

            BigDecimal result = service.toPln(new BigDecimal("10.00"), "EUR");

            assertThat(result).isEqualByComparingTo(new BigDecimal("43.00"));
        }
    }

    @Nested
    @DisplayName("Rounding")
    class Rounding {

        @Test
        @DisplayName("should round to 2 decimals using HALF_UP")
        void shouldRoundHalfUp() {
            when(exchangeRatePort.rateToPln("USD")).thenReturn(Optional.of(new BigDecimal("4.005")));

            BigDecimal result = service.toPln(new BigDecimal("10"), "USD");

            assertThat(result).isEqualByComparingTo(new BigDecimal("40.05"));
        }
    }

    @Nested
    @DisplayName("Missing rate")
    class MissingRate {

        @Test
        @DisplayName("should throw a translatable exception carrying the ISO code")
        void shouldThrowWhenRateMissing() {
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
}
