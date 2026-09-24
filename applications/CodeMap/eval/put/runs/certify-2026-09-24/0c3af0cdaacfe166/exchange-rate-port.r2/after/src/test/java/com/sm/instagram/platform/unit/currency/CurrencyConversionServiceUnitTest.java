package com.sm.instagram.platform.unit.currency;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.currency.CurrencyConversionService;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
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

    @Test
    @DisplayName("converts using the rate from the port, rounded to 2 decimals HALF_UP")
    void convertsUsingRate() {
        when(exchangeRatePort.rateToPln("USD")).thenReturn(Optional.of(new BigDecimal("3.955")));
        CurrencyConversionService service = new CurrencyConversionService(exchangeRatePort);

        BigDecimal result = service.toPln(new BigDecimal("10"), "USD");

        assertThat(result).isEqualByComparingTo("39.55");
    }

    @Test
    @DisplayName("no rate available throws a translatable exception with the ISO code")
    void missingRateThrows() {
        when(exchangeRatePort.rateToPln("XYZ")).thenReturn(Optional.empty());
        CurrencyConversionService service = new CurrencyConversionService(exchangeRatePort);

        assertThatThrownBy(() -> service.toPln(new BigDecimal("10"), "XYZ"))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> {
                    BusinessRuleTranslatableException translatable = (BusinessRuleTranslatableException) ex;
                    assertThat(translatable.getMessageKey()).isEqualTo("error.business.exchange_rate_unavailable");
                    assertThat(translatable.getArgs()).containsExactly("XYZ");
                });
    }
}
