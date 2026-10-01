package com.sm.instagram.platform.currency;

import com.sm.instagram.platform.common.exceptions.TranslatableException;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateProperties;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.catchThrowable;

/** Hidden acceptance test of the task exchange-rate-port (prompt under test, instance backend-conventions). */
class ExchangeRateAcceptanceTest {

    private static final String KEY = "error.business.exchange_rate_unavailable";

    @Test
    void thePortIsAnInterfaceAndTheAdapterImplementsIt() {
        assertThat(ExchangeRatePort.class.isInterface()).isTrue();
        assertThat(ExchangeRatePort.class.isAssignableFrom(FixedExchangeRateAdapter.class)).isTrue();
    }

    @Test
    void theFixedAdapterReadsTheConfiguredRates() {
        FixedExchangeRateProperties properties = new FixedExchangeRateProperties();
        Map<String, BigDecimal> rates = new HashMap<>();
        rates.put("EUR", new BigDecimal("4.30"));
        rates.put("usd", new BigDecimal("3.95"));
        properties.setRates(rates);
        FixedExchangeRateAdapter adapter = new FixedExchangeRateAdapter(properties);

        assertThat(adapter.rateToPln("EUR")).hasValueSatisfying(r -> assertThat(r).isEqualByComparingTo("4.30"));
        assertThat(adapter.rateToPln("eur")).hasValueSatisfying(r -> assertThat(r).isEqualByComparingTo("4.30"));
        assertThat(adapter.rateToPln("USD")).hasValueSatisfying(r -> assertThat(r).isEqualByComparingTo("3.95"));
        assertThat(adapter.rateToPln("PLN")).hasValueSatisfying(r -> assertThat(r).isEqualByComparingTo("1"));
        assertThat(adapter.rateToPln("GBP")).isEmpty();
    }

    @Test
    void theServiceConvertsThroughThePortAndRoundsHalfUp() {
        ExchangeRatePort port = iso -> "EUR".equals(iso) ? Optional.of(new BigDecimal("4.3055")) : Optional.empty();
        CurrencyConversionService service = new CurrencyConversionService(port);
        assertThat(service.toPln(new BigDecimal("10"), "EUR")).isEqualByComparingTo("43.06");
        assertThat(service.toPln(new BigDecimal("10"), "EUR").scale()).isEqualTo(2);
    }

    @Test
    void aMissingRateIsATranslatableErrorWithTheIsoCode() {
        CurrencyConversionService service = new CurrencyConversionService(iso -> Optional.empty());
        Throwable thrown = catchThrowable(() -> service.toPln(BigDecimal.ONE, "XYZ"));
        assertThat(thrown).isInstanceOf(TranslatableException.class);
        TranslatableException te = (TranslatableException) thrown;
        assertThat(te.getMessageKey()).isEqualTo(KEY);
        assertThat(te.getArgs()).contains("XYZ");
    }

    @Test
    void theKeyIsTranslatedInBothLanguages() throws IOException {
        assertThat(bundle("messages_en.properties").getProperty(KEY)).isNotBlank();
        assertThat(bundle("messages_pl.properties").getProperty(KEY)).isNotBlank();
    }

    private static Properties bundle(String name) throws IOException {
        Properties p = new Properties();
        try (InputStream in = ExchangeRateAcceptanceTest.class.getClassLoader().getResourceAsStream(name)) {
            assertThat(in).as(name).isNotNull();
            p.load(new InputStreamReader(in, StandardCharsets.UTF_8));
        }
        return p;
    }
}
