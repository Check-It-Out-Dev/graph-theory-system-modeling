package com.sm.instagram.platform.support.faq.services;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.common.exceptions.ResourceNotFoundException;
import com.sm.instagram.platform.common.util.RepositoryResolver;
import com.sm.instagram.platform.common.util.filtering.SpecificationBuilder;
import com.sm.instagram.platform.support.faq.models.Faq;
import com.sm.instagram.platform.support.faq.models.FaqCategory;
import com.sm.instagram.platform.support.faq.repositories.FaqCategoryRepository;
import com.sm.instagram.platform.support.faq.repositories.FaqRepository;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.assertj.core.groups.Tuple;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.modelmapper.ModelMapper;
import org.springframework.context.ApplicationContext;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link FaqService#reorderCategory(Long, List)}.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("FaqService reorderCategory")
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

    private FaqService service;

    @BeforeEach
    void setUp() {
        service = new FaqService(
                specificationBuilder,
                faqRepository,
                faqCategoryRepository,
                modelMapper,
                repositoryResolver,
                applicationContext);
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    private FaqCategory category(Long id) {
        FaqCategory category = new FaqCategory();
        category.setId(id);
        return category;
    }

    private Faq faq(Long id, FaqCategory category, int displayOrder) {
        Faq faq = new Faq();
        faq.setId(id);
        faq.setCategory(category);
        faq.setQuestion("Question " + id);
        faq.setAnswer("Answer " + id);
        faq.setDisplayOrder(displayOrder);
        faq.setActive(true);
        return faq;
    }

    @Test
    @DisplayName("assigns display order 1..n following the given id order and saves every FAQ")
    void reordersActiveFaqsAndSavesThem() {
        FaqCategory category = category(10L);
        Faq faq1 = faq(1L, category, 1);
        Faq faq2 = faq(2L, category, 2);
        Faq faq3 = faq(3L, category, 3);

        when(faqCategoryRepository.findById(10L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq1, faq2, faq3));
        when(applicationContext.getBean(FaqService.class)).thenReturn(service);
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken("admin-1", null, List.of()));
        when(faqRepository.save(any(Faq.class))).thenAnswer(invocation -> invocation.getArgument(0));

        service.reorderCategory(10L, List.of(3L, 1L, 2L));

        ArgumentCaptor<Faq> captor = ArgumentCaptor.forClass(Faq.class);
        verify(faqRepository, times(3)).save(captor.capture());
        List<Faq> saved = captor.getAllValues();

        assertThat(saved).extracting(Faq::getId, Faq::getDisplayOrder)
                .containsExactlyInAnyOrder(
                        Tuple.tuple(3L, 1),
                        Tuple.tuple(1L, 2),
                        Tuple.tuple(2L, 3));
    }

    @Test
    @DisplayName("rejects a list smaller than the category's active FAQs")
    void rejectsListMissingAnId() {
        FaqCategory category = category(10L);
        Faq faq1 = faq(1L, category, 1);
        Faq faq2 = faq(2L, category, 2);

        when(faqCategoryRepository.findById(10L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq1, faq2));

        assertThatThrownBy(() -> service.reorderCategory(10L, List.of(1L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> assertThat(((BusinessRuleTranslatableException) ex).getMessageKey())
                        .isEqualTo("error.business.faq_reorder_mismatch"));
        verify(faqRepository, never()).save(any());
    }

    @Test
    @DisplayName("rejects a list that repeats an id")
    void rejectsListWithDuplicateId() {
        FaqCategory category = category(10L);
        Faq faq1 = faq(1L, category, 1);
        Faq faq2 = faq(2L, category, 2);
        Faq faq3 = faq(3L, category, 3);

        when(faqCategoryRepository.findById(10L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq1, faq2, faq3));

        assertThatThrownBy(() -> service.reorderCategory(10L, List.of(1L, 1L, 2L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> assertThat(((BusinessRuleTranslatableException) ex).getMessageKey())
                        .isEqualTo("error.business.faq_reorder_mismatch"));
        verify(faqRepository, never()).save(any());
    }

    @Test
    @DisplayName("rejects a list containing an id from another category")
    void rejectsListWithForeignId() {
        FaqCategory category = category(10L);
        Faq faq1 = faq(1L, category, 1);
        Faq faq2 = faq(2L, category, 2);
        Faq faq3 = faq(3L, category, 3);

        when(faqCategoryRepository.findById(10L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq1, faq2, faq3));

        assertThatThrownBy(() -> service.reorderCategory(10L, List.of(1L, 2L, 99L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> assertThat(((BusinessRuleTranslatableException) ex).getMessageKey())
                        .isEqualTo("error.business.faq_reorder_mismatch"));
        verify(faqRepository, never()).save(any());
    }

    @Test
    @DisplayName("keeps the not-found behaviour for an unknown category")
    void keepsNotFoundBehaviourForUnknownCategory() {
        when(faqCategoryRepository.findById(999L)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.reorderCategory(999L, List.of(1L)))
                .isInstanceOf(ResourceNotFoundException.class);
        verify(faqRepository, never()).findByCategoryAndActiveTrueOrderByDisplayOrderAsc(any());
        verify(faqRepository, never()).save(any());
    }
}
