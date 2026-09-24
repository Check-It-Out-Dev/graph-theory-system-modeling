package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.currency.CurrencyConversionService;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateProperties;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@DisplayName("Currency Conversion Unit Tests")
class CurrencyConversionServiceUnitTest {

    @Mock
    private ExchangeRatePort exchangeRatePort;

    @Nested
    @DisplayName("CurrencyConversionService")
    class CurrencyConversionServiceTests {

        @Test
        @DisplayName("should convert amount to PLN using the port's rate, rounded HALF_UP to 2 decimals")
        void shouldConvertUsingPortRate() {
            when(exchangeRatePort.rateToPln("USD")).thenReturn(Optional.of(new BigDecimal("4.005")));
            CurrencyConversionService service = new CurrencyConversionService(exchangeRatePort);

            BigDecimal result = service.toPln(new BigDecimal("10"), "USD");

            assertThat(result).isEqualByComparingTo("40.05");
        }

        @Test
        @DisplayName("should throw a translatable exception with the ISO code when no rate exists")
        void shouldThrowWhenNoRate() {
            when(exchangeRatePort.rateToPln("XYZ")).thenReturn(Optional.empty());
            CurrencyConversionService service = new CurrencyConversionService(exchangeRatePort);

            assertThatThrownBy(() -> service.toPln(BigDecimal.TEN, "XYZ"))
                    .isInstanceOf(BusinessRuleTranslatableException.class)
                    .satisfies(ex -> {
                        BusinessRuleTranslatableException translatable = (BusinessRuleTranslatableException) ex;
                        assertThat(translatable.getMessageKey()).isEqualTo("error.business.exchange_rate_unavailable");
                        assertThat(translatable.getArgs()).containsExactly("XYZ");
                    });
        }
    }

    @Nested
    @DisplayName("FixedExchangeRateAdapter")
    class FixedExchangeRateAdapterTests {

        @Test
        @DisplayName("should match ISO codes case-insensitively and always return 1 for PLN")
        void shouldMatchCaseInsensitivelyAndDefaultPlnToOne() {
            FixedExchangeRateProperties properties = new FixedExchangeRateProperties();
            properties.setRates(Map.of("USD", new BigDecimal("4.00")));
            FixedExchangeRateAdapter adapter = new FixedExchangeRateAdapter(properties);

            assertThat(adapter.rateToPln("usd")).contains(new BigDecimal("4.00"));
            assertThat(adapter.rateToPln("PLN")).contains(BigDecimal.ONE);
            assertThat(adapter.rateToPln("pln")).contains(BigDecimal.ONE);
            assertThat(adapter.rateToPln("EUR")).isEmpty();
        }
    }
}
