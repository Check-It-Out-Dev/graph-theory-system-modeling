package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.common.exceptions.ResourceNotFoundException;
import com.sm.instagram.platform.common.util.RepositoryResolver;
import com.sm.instagram.platform.common.util.filtering.SpecificationBuilder;
import com.sm.instagram.platform.support.faq.models.Faq;
import com.sm.instagram.platform.support.faq.models.FaqCategory;
import com.sm.instagram.platform.support.faq.repositories.FaqCategoryRepository;
import com.sm.instagram.platform.support.faq.repositories.FaqRepository;
import com.sm.instagram.platform.support.faq.services.FaqService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.AfterEach;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.modelmapper.ModelMapper;
import org.springframework.context.ApplicationContext;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@DisplayName("FaqService Unit Tests")
class FaqServiceUnitTest {

    @Mock
    private SpecificationBuilder<Faq> specificationBuilder;

    @Mock
    private FaqRepository faqRepository;

    @Mock
    private FaqCategoryRepository faqCategoryRepository;

    @Mock
    private ModelMapper modelMapper;

    @Mock
    private RepositoryResolver repositoryResolver;

    @Mock
    private ApplicationContext applicationContext;

    private FaqService faqService;

    private FaqCategory category;
    private Faq faq1;
    private Faq faq2;
    private Faq faq3;

    @BeforeEach
    void setUp() {
        faqService = new FaqService(
                specificationBuilder,
                faqRepository,
                faqCategoryRepository,
                modelMapper,
                repositoryResolver,
                applicationContext);

        category = new FaqCategory();
        category.setId(1L);

        faq1 = new Faq();
        faq1.setId(10L);
        faq1.setCategory(category);
        faq1.setDisplayOrder(1);

        faq2 = new Faq();
        faq2.setId(20L);
        faq2.setCategory(category);
        faq2.setDisplayOrder(2);

        faq3 = new Faq();
        faq3.setId(30L);
        faq3.setCategory(category);
        faq3.setDisplayOrder(3);

        SecurityContext context = SecurityContextHolder.createEmptyContext();
        context.setAuthentication(new UsernamePasswordAuthenticationToken("admin", null, List.of()));
        SecurityContextHolder.setContext(context);
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    @Test
    @DisplayName("reorderCategory assigns display order i+1 following the requested id order")
    void reorderCategoryAssignsSequentialDisplayOrder() {
        when(applicationContext.getBean(FaqService.class)).thenReturn(faqService);
        when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq1, faq2, faq3));
        when(faqRepository.save(faq1)).thenReturn(faq1);
        when(faqRepository.save(faq2)).thenReturn(faq2);
        when(faqRepository.save(faq3)).thenReturn(faq3);

        faqService.reorderCategory(1L, List.of(30L, 10L, 20L));

        ArgumentCaptor<Faq> savedFaqCaptor = ArgumentCaptor.forClass(Faq.class);
        verify(faqRepository, times(3)).save(savedFaqCaptor.capture());

        List<Faq> saved = savedFaqCaptor.getAllValues();
        assertThat(saved).extracting(Faq::getId, Faq::getDisplayOrder)
                .containsExactly(
                        org.assertj.core.groups.Tuple.tuple(30L, 1),
                        org.assertj.core.groups.Tuple.tuple(10L, 2),
                        org.assertj.core.groups.Tuple.tuple(20L, 3));
    }

    @Test
    @DisplayName("reorderCategory throws a translatable mismatch error when an id is missing")
    void reorderCategoryThrowsWhenIdMissing() {
        when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq1, faq2, faq3));

        assertThatThrownBy(() -> faqService.reorderCategory(1L, List.of(10L, 20L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> assertThat(((BusinessRuleTranslatableException) ex).getMessageKey())
                        .isEqualTo("error.business.faq_reorder_mismatch"));

        verify(faqRepository, never()).save(org.mockito.ArgumentMatchers.any());
    }

    @Test
    @DisplayName("reorderCategory throws a translatable mismatch error when an id repeats")
    void reorderCategoryThrowsWhenIdRepeated() {
        when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq1, faq2, faq3));

        assertThatThrownBy(() -> faqService.reorderCategory(1L, List.of(10L, 10L, 30L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> assertThat(((BusinessRuleTranslatableException) ex).getMessageKey())
                        .isEqualTo("error.business.faq_reorder_mismatch"));

        verify(faqRepository, never()).save(org.mockito.ArgumentMatchers.any());
    }

    @Test
    @DisplayName("reorderCategory keeps not-found behaviour for an unknown category")
    void reorderCategoryThrowsNotFoundForUnknownCategory() {
        when(faqCategoryRepository.findById(99L)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> faqService.reorderCategory(99L, List.of(10L)))
                .isInstanceOf(ResourceNotFoundException.class);

        verify(faqRepository, never()).save(org.mockito.ArgumentMatchers.any());
    }
}
