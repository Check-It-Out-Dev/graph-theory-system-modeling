package com.sm.instagram.platform.currency;

import com.sm.instagram.platform.common.util.RepositoryResolver;
import com.sm.instagram.platform.common.util.filtering.SpecificationBuilder;
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
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

/**
 * Unit tests for CurrencyService's entity-to-DTO mapping paths, focused on
 * the displayOrder field carried from {@link Currency} to {@link CurrencyDtoOut}.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("CurrencyService Mapping Unit Tests")
class CurrencyServiceMappingUnitTest {

    @Mock
    private SpecificationBuilder<Currency> specificationBuilder;

    @Mock
    private CurrencyRepository currencyRepository;

    @Mock
    private RepositoryResolver repositoryResolver;

    @Mock
    private ApplicationContext applicationContext;

    private CurrencyService service;

    @BeforeEach
    void setUp() {
        service = new CurrencyService(
                specificationBuilder,
                currencyRepository,
                new ModelMapper(),
                repositoryResolver,
                applicationContext
        );
    }

    private Currency currency(String isoCode, int displayOrder) {
        Currency currency = new Currency();
        currency.setId(1L);
        currency.setName(isoCode);
        currency.setIsoCode(isoCode);
        currency.setSign("$");
        currency.setCountryCode("PL");
        currency.setDisplayOrder(displayOrder);
        return currency;
    }

    @Test
    @DisplayName("toDto carries displayOrder from the entity")
    void toDtoCarriesDisplayOrder() {
        Currency entity = currency("PLN", 0);

        CurrencyDtoOut dto = service.toDto(entity);

        assertThat(dto.getDisplayOrder()).isEqualTo(0);
    }

    @Test
    @DisplayName("getDataPagedAndFilteredAsDtos carries displayOrder for every entity, in page order")
    void getDataPagedAndFilteredAsDtosCarriesDisplayOrder() {
        Currency pln = currency("PLN", 0);
        Currency eur = currency("EUR", 1);
        Pageable pageable = Pageable.unpaged();
        when(currencyRepository.findAll((Specification<Currency>) null, pageable))
                .thenReturn(new PageImpl<>(List.of(pln, eur)));

        Page<CurrencyDtoOut> page = service.getDataPagedAndFilteredAsDtos(pageable, Map.of());

        List<CurrencyDtoOut> dtos = page.getContent();
        assertThat(dtos).hasSize(2);
        assertThat(dtos.get(0).getIsoCode()).isEqualTo("PLN");
        assertThat(dtos.get(0).getDisplayOrder()).isEqualTo(0);
        assertThat(dtos.get(1).getIsoCode()).isEqualTo("EUR");
        assertThat(dtos.get(1).getDisplayOrder()).isEqualTo(1);
    }
}
