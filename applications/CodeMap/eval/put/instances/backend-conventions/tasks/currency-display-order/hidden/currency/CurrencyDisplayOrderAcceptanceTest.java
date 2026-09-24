package com.sm.instagram.platform.currency;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.Spy;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.modelmapper.ModelMapper;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.PageRequest;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;

/** Hidden acceptance test of the task currency-display-order (prompt under test, instance backend-conventions). */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class CurrencyDisplayOrderAcceptanceTest {

    @Mock CurrencyRepository currencyRepository;
    @Spy ModelMapper modelMapper = new ModelMapper();
    @InjectMocks CurrencyService service;

    @Test
    void theEntityStoresTheDisplayOrder() throws Exception {
        Field field = Currency.class.getDeclaredField("displayOrder");
        assertThat(field.getType()).isIn(int.class, Integer.class);
        Currency c = currency(4);
        assertThat(c.getDisplayOrder()).isEqualTo(4);
    }

    @Test
    void toDtoCarriesTheDisplayOrder() {
        CurrencyDtoOut dto = service.toDto(currency(4));
        assertThat(dto.getDisplayOrder()).isEqualTo(4);
    }

    @Test
    void thePagedListCarriesTheDisplayOrder() {
        CurrencyService spy = Mockito.spy(service);
        Page<Currency> page = new PageImpl<>(List.of(currency(1), currency(2)));
        doReturn(page).when(spy).getDataPagedAndFiltered(any(), any());
        Page<CurrencyDtoOut> dtos = spy.getDataPagedAndFilteredAsDtos(PageRequest.of(0, 10), Map.of());
        assertThat(dtos.getContent()).extracting(CurrencyDtoOut::getDisplayOrder).containsExactly(1, 2);
    }

    @Test
    void aChangesetInTheMasterChangelogAddsTheColumn() throws IOException {
        Pattern sql = Pattern.compile("(?is)alter\\s+table\\s+(public\\.)?\"?currency\"?\\s+add\\s+(column\\s+)?(if\\s+not\\s+exists\\s+)?\"?display_order\"?\\b");
        Pattern xml = Pattern.compile("(?is)<addColumn[^>]*tableName=\"currency\"[^>]*>.*?<column[^>]*name=\"display_order\"");
        boolean found = false;
        for (String file : includedChangesets()) {
            String text = resource("db/changelog/" + file);
            if (text != null && (sql.matcher(text).find() || xml.matcher(text).find())) {
                found = true;
            }
        }
        assertThat(found).as("an included changeset adds currency.display_order").isTrue();
    }

    private static Currency currency(int order) {
        Currency c = new Currency();
        c.setId((long) order);
        c.setName("Currency " + order);
        c.setIsoCode("C" + order);
        c.setSign("$");
        c.setCountryCode("XX");
        c.setDisplayOrder(order);
        return c;
    }

    private static List<String> includedChangesets() throws IOException {
        String master = resource("db/changelog/changelog.xml");
        assertThat(master).isNotNull();
        List<String> files = new ArrayList<>();
        Matcher m = Pattern.compile("<include\\s+file=\"([^\"]+)\"").matcher(master.replaceAll("(?s)<!--.*?-->", ""));
        while (m.find()) {
            files.add(m.group(1));
        }
        return files;
    }

    private static String resource(String path) throws IOException {
        try (InputStream in = CurrencyDisplayOrderAcceptanceTest.class.getClassLoader().getResourceAsStream(path)) {
            return in == null ? null : new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
