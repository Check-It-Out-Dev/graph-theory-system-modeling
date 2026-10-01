package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.currency.Currency;
import com.sm.instagram.platform.currency.CurrencyDtoOut;
import com.sm.instagram.platform.currency.CurrencyRepository;
import com.sm.instagram.platform.currency.CurrencyService;
import com.sm.instagram.platform.common.util.RepositoryResolver;
import com.sm.instagram.platform.common.util.filtering.SpecificationBuilder;
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
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;

import java.util.Collections;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Unit tests for CurrencyService's DTO mapping of the currency display order.
 */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
@DisplayName("CurrencyService displayOrder mapping")
class CurrencyServiceDisplayOrderUnitTest {

    @Mock
    private SpecificationBuilder<Currency> specificationBuilder;

    @Mock
    private CurrencyRepository currencyRepository;

    @Mock
    private ModelMapper modelMapper;

    @Mock
    private RepositoryResolver repositoryResolver;

    @Mock
    private ApplicationContext applicationContext;

    private CurrencyService currencyService;

    @BeforeEach
    void setUp() {
        // Anonymous subclass to access the protected constructor.
        currencyService = new CurrencyService(
                specificationBuilder,
                currencyRepository,
                modelMapper,
                repositoryResolver,
                applicationContext
        ) {};
    }

    private Currency pln() {
        return new Currency(1L, "Polish Zloty", "PLN", "zł", "PL", 0);
    }

    @Test
    @DisplayName("toDto maps displayOrder onto CurrencyDtoOut")
    void toDtoMapsDisplayOrder() {
        Currency entity = pln();
        CurrencyDtoOut expected = CurrencyDtoOut.builder()
                .id(entity.getId())
                .name(entity.getName())
                .isoCode(entity.getIsoCode())
                .sign(entity.getSign())
                .countryCode(entity.getCountryCode())
                .displayOrder(entity.getDisplayOrder())
                .build();
        when(modelMapper.map(entity, CurrencyDtoOut.class)).thenReturn(expected);

        CurrencyDtoOut dto = currencyService.toDto(entity);

        assertThat(dto.getDisplayOrder()).isEqualTo(0);
    }

    @Test
    @DisplayName("getDataPagedAndFilteredAsDtos maps displayOrder onto each CurrencyDtoOut")
    void getDataPagedAndFilteredAsDtosMapsDisplayOrder() {
        Currency entity = pln();
        Page<Currency> page = new PageImpl<>(Collections.singletonList(entity));
        when(specificationBuilder.createSpecification(any())).thenReturn((Specification) mock(Specification.class));
        when(currencyRepository.findAll(any(Specification.class), any(Pageable.class))).thenReturn(page);

        Page<CurrencyDtoOut> dtoPage = currencyService.getDataPagedAndFilteredAsDtos(
                Pageable.unpaged(), Collections.<String, String>emptyMap());

        assertThat(dtoPage.getContent()).hasSize(1);
        assertThat(dtoPage.getContent().get(0).getDisplayOrder()).isEqualTo(0);
    }
}
