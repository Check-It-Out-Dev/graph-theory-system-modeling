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
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

/**
 * Unit tests for CurrencyService's DTO mapping paths, focused on the display_order field.
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

    private CurrencyService currencyService;

    @BeforeEach
    void setUp() {
        currencyService = new CurrencyService(
                specificationBuilder,
                currencyRepository,
                new ModelMapper(),
                repositoryResolver,
                applicationContext
        );
    }

    @Test
    @DisplayName("toDto should carry the entity's displayOrder into the DTO")
    void toDtoShouldCarryDisplayOrder() {
        Currency entity = new Currency();
        entity.setId(7L);
        entity.setName("Polish Zloty");
        entity.setIsoCode("PLN");
        entity.setSign("zł");
        entity.setCountryCode("PL");
        entity.setDisplayOrder(1);

        CurrencyDtoOut dto = currencyService.toDto(entity);

        assertThat(dto.getDisplayOrder()).isEqualTo(1);
        assertThat(dto.getIsoCode()).isEqualTo("PLN");
    }

    @Test
    @DisplayName("getDataPagedAndFilteredAsDtos should carry each entity's displayOrder into its DTO")
    void getDataPagedAndFilteredAsDtosShouldCarryDisplayOrder() {
        Currency pln = new Currency();
        pln.setId(1L);
        pln.setName("Polish Zloty");
        pln.setIsoCode("PLN");
        pln.setSign("zł");
        pln.setCountryCode("PL");
        pln.setDisplayOrder(1);

        Currency eur = new Currency();
        eur.setId(2L);
        eur.setName("Euro");
        eur.setIsoCode("EUR");
        eur.setSign("€");
        eur.setCountryCode("DE");
        eur.setDisplayOrder(2);

        Map<String, String> filters = Map.of();
        Pageable pageable = PageRequest.of(0, 10);
        Specification<Currency> spec = (root, query, cb) -> null;

        when(specificationBuilder.createSpecification(filters)).thenReturn(spec);
        when(currencyRepository.findAll(spec, pageable))
                .thenReturn(new PageImpl<>(List.of(pln, eur)));

        Page<CurrencyDtoOut> result = currencyService.getDataPagedAndFilteredAsDtos(pageable, filters);

        assertThat(result.getContent()).hasSize(2);
        assertThat(result.getContent().get(0).getIsoCode()).isEqualTo("PLN");
        assertThat(result.getContent().get(0).getDisplayOrder()).isEqualTo(1);
        assertThat(result.getContent().get(1).getIsoCode()).isEqualTo("EUR");
        assertThat(result.getContent().get(1).getDisplayOrder()).isEqualTo(2);
    }
}
