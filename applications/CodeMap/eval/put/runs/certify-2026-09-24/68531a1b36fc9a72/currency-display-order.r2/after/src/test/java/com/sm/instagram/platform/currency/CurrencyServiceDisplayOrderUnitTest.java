package com.sm.instagram.platform.currency;

import com.sm.instagram.platform.common.util.RepositoryResolver;
import com.sm.instagram.platform.common.util.filtering.SpecificationBuilder;
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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class CurrencyServiceDisplayOrderUnitTest {

    @Mock
    private SpecificationBuilder<Currency> specificationBuilder;
    @Mock
    private CurrencyRepository currencyRepository;
    @Mock
    private RepositoryResolver repositoryResolver;
    @Mock
    private ApplicationContext applicationContext;

    private CurrencyService buildService() {
        return new CurrencyService(specificationBuilder, currencyRepository, new ModelMapper(), repositoryResolver, applicationContext);
    }

    @Test
    void currencyStoresDisplayOrderDistinctFromDefault() {
        Currency currency = new Currency();

        assertThat(currency.getDisplayOrder()).isEqualTo(0);

        currency.setDisplayOrder(1);

        assertThat(currency.getDisplayOrder()).isEqualTo(1);
    }

    @Test
    void toDtoMapsDisplayOrderFromEntity() {
        CurrencyService service = buildService();
        Currency pln = new Currency(1L, "Polish Zloty", "PLN", "zl", "PL", 1);

        CurrencyDtoOut dto = service.toDto(pln);

        assertThat(dto.getDisplayOrder()).isEqualTo(1);
        assertThat(dto.getIsoCode()).isEqualTo("PLN");
    }

    @Test
    void getDataPagedAndFilteredAsDtosMapsDisplayOrderForEveryRow() {
        CurrencyService service = buildService();
        Currency pln = new Currency(1L, "Polish Zloty", "PLN", "zl", "PL", 1);
        Currency eur = new Currency(2L, "Euro", "EUR", "e", "EU", 2);
        Pageable pageable = PageRequest.of(0, 10);
        Page<Currency> entityPage = new PageImpl<>(List.of(pln, eur), pageable, 2);

        when(specificationBuilder.createSpecification(any())).thenReturn(mockSpecification());
        when(currencyRepository.findAll(any(Specification.class), eq(pageable))).thenReturn(entityPage);

        Page<CurrencyDtoOut> result = service.getDataPagedAndFilteredAsDtos(pageable, Map.of());

        assertThat(result.getContent()).hasSize(2);
        assertThat(result.getContent().get(0).getIsoCode()).isEqualTo("PLN");
        assertThat(result.getContent().get(0).getDisplayOrder()).isEqualTo(1);
        assertThat(result.getContent().get(1).getIsoCode()).isEqualTo("EUR");
        assertThat(result.getContent().get(1).getDisplayOrder()).isEqualTo(2);
    }

    private Specification<Currency> mockSpecification() {
        return (root, query, cb) -> null;
    }
}
