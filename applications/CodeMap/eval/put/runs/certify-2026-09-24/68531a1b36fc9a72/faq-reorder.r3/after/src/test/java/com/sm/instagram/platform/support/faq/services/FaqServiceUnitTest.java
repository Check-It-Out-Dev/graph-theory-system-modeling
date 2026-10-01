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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@DisplayName("FaqService Unit Tests")
class FaqServiceUnitTest {

    private static final Long CATEGORY_ID = 7L;

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
    private FaqCategory category;
    private Faq faq10;
    private Faq faq20;
    private Faq faq30;

    @BeforeEach
    void setUp() {
        service = new FaqService(
                specificationBuilder,
                faqRepository,
                faqCategoryRepository,
                modelMapper,
                repositoryResolver,
                applicationContext
        );

        category = new FaqCategory();
        category.setId(CATEGORY_ID);
        category.setName("Account");

        faq10 = buildFaq(10L, category, 3);
        faq20 = buildFaq(20L, category, 1);
        faq30 = buildFaq(30L, category, 2);
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    private Faq buildFaq(Long id, FaqCategory owner, int displayOrder) {
        Faq faq = new Faq();
        faq.setId(id);
        faq.setCategory(owner);
        faq.setQuestion("Question " + id);
        faq.setAnswer("Answer " + id);
        faq.setDisplayOrder(displayOrder);
        faq.setActive(true);
        return faq;
    }

    private void authenticate() {
        SecurityContext context = SecurityContextHolder.createEmptyContext();
        context.setAuthentication(new UsernamePasswordAuthenticationToken("admin-firebase-uid", null, List.of()));
        SecurityContextHolder.setContext(context);
        when(applicationContext.getBean(FaqService.class)).thenReturn(service);
    }

    @Test
    @DisplayName("assigns sequential display orders in the requested order and saves each FAQ")
    void reorderCategoryAssignsSequentialDisplayOrders() {
        when(faqCategoryRepository.findById(CATEGORY_ID)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq20, faq30, faq10));
        authenticate();

        service.reorderCategory(CATEGORY_ID, List.of(30L, 10L, 20L));

        ArgumentCaptor<Faq> captor = ArgumentCaptor.forClass(Faq.class);
        verify(faqRepository, times(3)).save(captor.capture());

        assertThat(captor.getAllValues())
                .extracting(Faq::getId, Faq::getDisplayOrder)
                .containsExactly(
                        org.assertj.core.groups.Tuple.tuple(30L, 1),
                        org.assertj.core.groups.Tuple.tuple(10L, 2),
                        org.assertj.core.groups.Tuple.tuple(20L, 3)
                );
    }

    @Test
    @DisplayName("keeps the not-found behaviour for an unknown category")
    void reorderCategoryUnknownCategoryThrowsNotFound() {
        when(faqCategoryRepository.findById(CATEGORY_ID)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.reorderCategory(CATEGORY_ID, List.of(10L)))
                .isInstanceOf(ResourceNotFoundException.class)
                .extracting(ex -> ((ResourceNotFoundException) ex).getMessageKey())
                .isEqualTo("error.business.item_not_found");

        verify(faqRepository, never()).findByCategoryAndActiveTrueOrderByDisplayOrderAsc(any());
        verify(faqRepository, never()).save(any());
    }

    @Test
    @DisplayName("rejects a list missing one of the category's active FAQs")
    void reorderCategoryMissingIdThrowsMismatch() {
        when(faqCategoryRepository.findById(CATEGORY_ID)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq20, faq30, faq10));

        assertThatThrownBy(() -> service.reorderCategory(CATEGORY_ID, List.of(10L, 20L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .extracting(ex -> ((BusinessRuleTranslatableException) ex).getMessageKey())
                .isEqualTo("error.business.faq_reorder_mismatch");

        verify(faqRepository, never()).save(any());
    }

    @Test
    @DisplayName("rejects a list that repeats an id instead of listing each FAQ once")
    void reorderCategoryDuplicateIdThrowsMismatch() {
        when(faqCategoryRepository.findById(CATEGORY_ID)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq20, faq30, faq10));

        assertThatThrownBy(() -> service.reorderCategory(CATEGORY_ID, List.of(10L, 10L, 20L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .extracting(ex -> ((BusinessRuleTranslatableException) ex).getMessageKey())
                .isEqualTo("error.business.faq_reorder_mismatch");

        verify(faqRepository, never()).save(any());
    }

    @Test
    @DisplayName("rejects a list containing an id that does not belong to the category's active FAQs")
    void reorderCategoryForeignIdThrowsMismatch() {
        when(faqCategoryRepository.findById(CATEGORY_ID)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq20, faq30, faq10));

        assertThatThrownBy(() -> service.reorderCategory(CATEGORY_ID, List.of(10L, 20L, 999L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .extracting(ex -> ((BusinessRuleTranslatableException) ex).getMessageKey())
                .isEqualTo("error.business.faq_reorder_mismatch");

        verify(faqRepository, never()).save(any());
    }

    @Test
    @DisplayName("rejects a null list of ids")
    void reorderCategoryNullListThrowsMismatch() {
        when(faqCategoryRepository.findById(CATEGORY_ID)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq20, faq30, faq10));

        assertThatThrownBy(() -> service.reorderCategory(CATEGORY_ID, null))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .extracting(ex -> ((BusinessRuleTranslatableException) ex).getMessageKey())
                .isEqualTo("error.business.faq_reorder_mismatch");

        verify(faqRepository, never()).save(any());
    }
}
