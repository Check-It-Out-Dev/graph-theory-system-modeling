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
 * Unit tests for the currency {@code displayOrder} mapping added to
 * {@link CurrencyService#toDto} and {@link CurrencyService#getDataPagedAndFilteredAsDtos}
 * so PLN can be shown first instead of alphabetically.
 */
@ExtendWith(MockitoExtension.class)
class CurrencyServiceDisplayOrderUnitTest {

    @Mock private SpecificationBuilder<Currency> specificationBuilder;
    @Mock private CurrencyRepository currencyRepository;
    @Mock private RepositoryResolver repositoryResolver;
    @Mock private ApplicationContext applicationContext;

    private CurrencyService service;

    @BeforeEach
    void setUp() {
        // A real ModelMapper (no Spring context) exercises the actual field-name mapping.
        service = new CurrencyService(specificationBuilder, currencyRepository,
                new ModelMapper(), repositoryResolver, applicationContext);
    }

    private Currency currency(Long id, String isoCode, int displayOrder) {
        Currency currency = new Currency();
        currency.setId(id);
        currency.setName(isoCode + " name");
        currency.setIsoCode(isoCode);
        currency.setSign("z");
        currency.setCountryCode("XX");
        currency.setDisplayOrder(displayOrder);
        return currency;
    }

    @Test
    @DisplayName("toDto carries the entity's displayOrder onto the DTO")
    void toDtoMapsDisplayOrder() {
        Currency entity = currency(1L, "PLN", 1);

        CurrencyDtoOut dto = service.toDto(entity);

        assertThat(dto.getDisplayOrder()).isEqualTo(1);
        assertThat(dto.getIsoCode()).isEqualTo("PLN");
    }

    @Test
    @DisplayName("getDataPagedAndFilteredAsDtos maps displayOrder for every row, PLN first")
    void pagedDtosMapDisplayOrderForEveryRow() {
        Currency pln = currency(1L, "PLN", 1);
        Currency eur = currency(2L, "EUR", 2);
        Currency usd = currency(3L, "USD", 3);
        Pageable pageable = PageRequest.of(0, 10);
        Map<String, String> filters = Map.of();
        Specification<Currency> spec = (root, query, cb) -> null;

        when(specificationBuilder.createSpecification(filters)).thenReturn(spec);
        when(currencyRepository.findAll(spec, pageable))
                .thenReturn(new PageImpl<>(List.of(pln, eur, usd), pageable, 3));

        Page<CurrencyDtoOut> result = service.getDataPagedAndFilteredAsDtos(pageable, filters);

        assertThat(result.getContent())
                .extracting(CurrencyDtoOut::getIsoCode, CurrencyDtoOut::getDisplayOrder)
                .containsExactly(
                        org.assertj.core.groups.Tuple.tuple("PLN", 1),
                        org.assertj.core.groups.Tuple.tuple("EUR", 2),
                        org.assertj.core.groups.Tuple.tuple("USD", 3));
    }

    @Test
    @DisplayName("getDataPagedAndFilteredAsDtos returns an empty page when there are no currencies")
    void pagedDtosHandleEmptyPage() {
        Pageable pageable = PageRequest.of(0, 10);
        Map<String, String> filters = Map.of();
        Specification<Currency> spec = (root, query, cb) -> null;

        when(specificationBuilder.createSpecification(filters)).thenReturn(spec);
        when(currencyRepository.findAll(spec, pageable))
                .thenReturn(new PageImpl<>(List.of(), pageable, 0));

        Page<CurrencyDtoOut> result = service.getDataPagedAndFilteredAsDtos(pageable, filters);

        assertThat(result.getContent()).isEmpty();
        assertThat(result.getTotalElements()).isZero();
    }
}
