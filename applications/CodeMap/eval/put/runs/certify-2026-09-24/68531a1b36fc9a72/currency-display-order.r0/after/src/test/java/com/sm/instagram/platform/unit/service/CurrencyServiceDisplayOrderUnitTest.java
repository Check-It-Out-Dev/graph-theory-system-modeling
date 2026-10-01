package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.util.RepositoryResolver;
import com.sm.instagram.platform.common.util.filtering.SpecificationBuilder;
import com.sm.instagram.platform.currency.Currency;
import com.sm.instagram.platform.currency.CurrencyDtoOut;
import com.sm.instagram.platform.currency.CurrencyRepository;
import com.sm.instagram.platform.currency.CurrencyService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.modelmapper.ModelMapper;
import org.springframework.context.ApplicationContext;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;

import java.lang.reflect.Constructor;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Unit tests for the two CurrencyService mapping paths that carry a
 * currency's team-chosen display order (PLN first) to clients.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("CurrencyService display order mapping")
class CurrencyServiceDisplayOrderUnitTest {

    @Mock
    private SpecificationBuilder<Currency> specificationBuilder;

    @Mock
    private CurrencyRepository currencyRepository;

    @Mock
    private RepositoryResolver repositoryResolver;

    @Mock
    private ApplicationContext applicationContext;

    private CurrencyService currencyService;

    @BeforeEach
    void setUp() throws Exception {
        Constructor<CurrencyService> constructor = CurrencyService.class.getDeclaredConstructor(
                SpecificationBuilder.class, CurrencyRepository.class, ModelMapper.class,
                RepositoryResolver.class, ApplicationContext.class);
        constructor.setAccessible(true);
        currencyService = constructor.newInstance(
                specificationBuilder, currencyRepository, new ModelMapper(), repositoryResolver, applicationContext);
    }

    @Test
    @DisplayName("a currency defaults to display order 0, matching the column default")
    void newCurrencyDefaultsToZeroDisplayOrder() {
        Currency currency = new Currency();

        assertThat(currency.getDisplayOrder()).isZero();
    }

    @Test
    @DisplayName("toDto carries the entity's display order")
    void toDtoCarriesDisplayOrder() {
        Currency pln = new Currency(1L, "Polish Zloty", "PLN", "zl", "PL", 1);

        CurrencyDtoOut dto = currencyService.toDto(pln);

        assertThat(dto.getIsoCode()).isEqualTo("PLN");
        assertThat(dto.getDisplayOrder()).isEqualTo(1);
    }

    @Test
    @DisplayName("getDataPagedAndFilteredAsDtos carries each entity's own display order, PLN first")
    void getDataPagedAndFilteredAsDtosCarriesDisplayOrderPerEntity() {
        Currency eur = new Currency(2L, "Euro", "EUR", "€", "DE", 2);
        Currency pln = new Currency(1L, "Polish Zloty", "PLN", "zl", "PL", 1);
        Currency usd = new Currency(3L, "US Dollar", "USD", "$", "US", 3);
        Page<Currency> repositoryPage = new PageImpl<>(List.of(eur, pln, usd));
        Pageable pageable = PageRequest.of(0, 10);
        Map<String, String> filters = new HashMap<>();
        Specification<Currency> specification = mock(Specification.class);

        when(specificationBuilder.createSpecification(filters)).thenReturn(specification);
        when(currencyRepository.findAll(eq(specification), any(Pageable.class))).thenReturn(repositoryPage);

        Page<CurrencyDtoOut> result = currencyService.getDataPagedAndFilteredAsDtos(pageable, filters);

        Map<String, Integer> displayOrderByIsoCode = new HashMap<>();
        for (CurrencyDtoOut dto : result.getContent()) {
            displayOrderByIsoCode.put(dto.getIsoCode(), dto.getDisplayOrder());
        }

        assertThat(displayOrderByIsoCode)
                .containsEntry("PLN", 1)
                .containsEntry("EUR", 2)
                .containsEntry("USD", 3);
        assertThat(displayOrderByIsoCode.get("PLN")).isLessThan(displayOrderByIsoCode.get("EUR"));
        assertThat(displayOrderByIsoCode.get("PLN")).isLessThan(displayOrderByIsoCode.get("USD"));
    }
}
