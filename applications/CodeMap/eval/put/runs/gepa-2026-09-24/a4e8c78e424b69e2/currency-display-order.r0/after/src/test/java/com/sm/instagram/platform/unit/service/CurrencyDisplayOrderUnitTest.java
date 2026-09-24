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
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.modelmapper.ModelMapper;
import org.springframework.context.ApplicationContext;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;

import java.lang.reflect.Constructor;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Unit tests for the currency display order: both mapping paths of
 * CurrencyService must carry it from the entity to CurrencyDtoOut so clients
 * can order currencies the way the team chose (PLN first) instead of alphabetically.
 */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
@DisplayName("Currency displayOrder Unit Tests")
class CurrencyDisplayOrderUnitTest {

    @Mock
    private SpecificationBuilder<Currency> specificationBuilder;

    @Mock
    private CurrencyRepository currencyRepository;

    @Mock
    private RepositoryResolver repositoryResolver;

    @Mock
    private ApplicationContext applicationContext;

    private CurrencyService service;

    private Currency pln;
    private Currency eur;

    @BeforeEach
    void setUp() throws ReflectiveOperationException {
        // CurrencyService's constructor is protected (constructor_injection convention),
        // so a test outside its package reaches it reflectively.
        Constructor<CurrencyService> constructor = CurrencyService.class.getDeclaredConstructor(
                SpecificationBuilder.class, CurrencyRepository.class, ModelMapper.class,
                RepositoryResolver.class, ApplicationContext.class);
        constructor.setAccessible(true);
        service = constructor.newInstance(specificationBuilder, currencyRepository, new ModelMapper(),
                repositoryResolver, applicationContext);

        pln = new Currency(1L, "Polish Zloty", "PLN", "zl", "PL", 1);
        eur = new Currency(2L, "Euro", "EUR", "EUR", "DE", 2);
    }

    @Test
    @DisplayName("toDto copies displayOrder from entity to CurrencyDtoOut")
    void toDtoCopiesDisplayOrder() {
        CurrencyDtoOut dto = service.toDto(pln);

        assertThat(dto.getDisplayOrder()).isEqualTo(1);
    }

    @Test
    @DisplayName("getDataPagedAndFilteredAsDtos copies displayOrder for every row")
    void getDataPagedAndFilteredAsDtosCopiesDisplayOrder() {
        Pageable pageable = PageRequest.of(0, 10);
        Page<Currency> page = new PageImpl<>(List.of(pln, eur), pageable, 2);
        when(specificationBuilder.createSpecification(anyMap())).thenReturn(mock(Specification.class));
        when(currencyRepository.findAll(any(Specification.class), any(Pageable.class))).thenReturn(page);

        Page<CurrencyDtoOut> result = service.getDataPagedAndFilteredAsDtos(pageable, Map.of());

        assertThat(result.getContent()).extracting(CurrencyDtoOut::getDisplayOrder).containsExactly(1, 2);
    }

    @Test
    @DisplayName("PLN's lower displayOrder puts it ahead of currencies seeded after it")
    void plnDisplayOrderPutsItFirst() {
        Pageable pageable = PageRequest.of(0, 10);
        Page<Currency> page = new PageImpl<>(List.of(pln, eur), pageable, 2);
        when(specificationBuilder.createSpecification(anyMap())).thenReturn(mock(Specification.class));
        when(currencyRepository.findAll(any(Specification.class), any(Pageable.class))).thenReturn(page);

        Page<CurrencyDtoOut> result = service.getDataPagedAndFilteredAsDtos(pageable, Map.of());

        List<CurrencyDtoOut> sorted = result.getContent().stream()
                .sorted((a, b) -> Integer.compare(a.getDisplayOrder(), b.getDisplayOrder()))
                .toList();
        assertThat(sorted.get(0).getIsoCode()).isEqualTo("PLN");
    }
}
