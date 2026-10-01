package com.sm.instagram.platform.currency;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
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

    @Test
    void shouldConvertAmountUsingRateFromPort() {
        when(exchangeRatePort.rateToPln("EUR")).thenReturn(Optional.of(new BigDecimal("4.30")));
        CurrencyConversionService service = new CurrencyConversionService(exchangeRatePort);

        BigDecimal result = service.toPln(new BigDecimal("10.00"), "EUR");

        assertThat(result).isEqualByComparingTo("43.00");
    }

    @Test
    void shouldRoundHalfUpToTwoDecimals() {
        when(exchangeRatePort.rateToPln("USD")).thenReturn(Optional.of(new BigDecimal("3.995")));
        CurrencyConversionService service = new CurrencyConversionService(exchangeRatePort);

        BigDecimal result = service.toPln(new BigDecimal("2"), "USD");

        assertThat(result).isEqualByComparingTo("7.99");
    }

    @Test
    void shouldPassIsoCodeThroughToThePortUnchanged() {
        when(exchangeRatePort.rateToPln("gbp")).thenReturn(Optional.of(BigDecimal.ONE));
        CurrencyConversionService service = new CurrencyConversionService(exchangeRatePort);

        service.toPln(BigDecimal.TEN, "gbp");

        ArgumentCaptor<String> isoCodeCaptor = ArgumentCaptor.forClass(String.class);
        verify(exchangeRatePort).rateToPln(isoCodeCaptor.capture());
        assertThat(isoCodeCaptor.getValue()).isEqualTo("gbp");
    }

    @Test
    void shouldThrowTranslatableExceptionWithIsoCodeWhenRateMissing() {
        when(exchangeRatePort.rateToPln("XYZ")).thenReturn(Optional.empty());
        CurrencyConversionService service = new CurrencyConversionService(exchangeRatePort);

        assertThatThrownBy(() -> service.toPln(new BigDecimal("10.00"), "XYZ"))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> {
                    BusinessRuleTranslatableException translatable = (BusinessRuleTranslatableException) ex;
                    assertThat(translatable.getMessageKey()).isEqualTo("error.business.exchange_rate_unavailable");
                    assertThat(translatable.getArgs()).containsExactly("XYZ");
                });
    }
}
