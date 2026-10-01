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
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.modelmapper.ModelMapper;
import org.springframework.context.ApplicationContext;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link FaqService#reorderCategory(Long, List)}.
 */
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

    private FaqService service;

    private FaqCategory category;
    private Faq faqOne;
    private Faq faqTwo;
    private Faq faqThree;

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
        category.setId(7L);

        faqOne = new Faq();
        faqOne.setId(10L);
        faqOne.setCategory(category);
        faqOne.setDisplayOrder(3);

        faqTwo = new Faq();
        faqTwo.setId(20L);
        faqTwo.setCategory(category);
        faqTwo.setDisplayOrder(1);

        faqThree = new Faq();
        faqThree.setId(30L);
        faqThree.setCategory(category);
        faqThree.setDisplayOrder(2);
    }

    private void stubSecurityContextForSave() {
        SecurityContext securityContext = org.mockito.Mockito.mock(SecurityContext.class);
        Authentication authentication = org.mockito.Mockito.mock(Authentication.class);
        SecurityContextHolder.setContext(securityContext);
        when(securityContext.getAuthentication()).thenReturn(authentication);
        when(authentication.getName()).thenReturn("admin-1");
        when(applicationContext.getBean(FaqService.class)).thenReturn(service);
    }

    @Test
    @DisplayName("assigns display order 1..n following the given id order and saves each FAQ")
    void reordersFaqsAccordingToGivenIds() {
        // Given
        when(faqCategoryRepository.findById(7L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faqTwo, faqThree, faqOne));
        when(faqRepository.save(any(Faq.class))).thenAnswer(invocation -> invocation.getArgument(0));
        stubSecurityContextForSave();

        // When
        service.reorderCategory(7L, List.of(30L, 10L, 20L));

        // Then
        ArgumentCaptor<Faq> captor = ArgumentCaptor.forClass(Faq.class);
        verify(faqRepository, org.mockito.Mockito.times(3)).save(captor.capture());
        List<Faq> saved = captor.getAllValues();

        assertThat(faqThree.getDisplayOrder()).isEqualTo(1);
        assertThat(faqOne.getDisplayOrder()).isEqualTo(2);
        assertThat(faqTwo.getDisplayOrder()).isEqualTo(3);
        assertThat(saved).extracting(Faq::getId).containsExactly(30L, 10L, 20L);
    }

    @Test
    @DisplayName("rejects a list missing one of the category's active FAQs")
    void rejectsListMissingAFaq() {
        // Given
        when(faqCategoryRepository.findById(7L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faqOne, faqTwo, faqThree));

        // When / Then
        assertThatThrownBy(() -> service.reorderCategory(7L, List.of(10L, 20L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .extracting(ex -> ((BusinessRuleTranslatableException) ex).getMessageKey())
                .isEqualTo("error.business.faq_reorder_mismatch");
        verify(faqRepository, never()).save(any());
    }

    @Test
    @DisplayName("rejects a list with a duplicated id even when the size matches")
    void rejectsListWithDuplicateId() {
        // Given
        when(faqCategoryRepository.findById(7L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faqOne, faqTwo, faqThree));

        // When / Then
        assertThatThrownBy(() -> service.reorderCategory(7L, List.of(10L, 10L, 20L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .extracting(ex -> ((BusinessRuleTranslatableException) ex).getMessageKey())
                .isEqualTo("error.business.faq_reorder_mismatch");
        verify(faqRepository, never()).save(any());
    }

    @Test
    @DisplayName("rejects an id that belongs to another category")
    void rejectsIdFromAnotherCategory() {
        // Given
        when(faqCategoryRepository.findById(7L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faqOne, faqTwo, faqThree));

        // When / Then
        assertThatThrownBy(() -> service.reorderCategory(7L, List.of(10L, 20L, 999L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .extracting(ex -> ((BusinessRuleTranslatableException) ex).getMessageKey())
                .isEqualTo("error.business.faq_reorder_mismatch");
        verify(faqRepository, never()).save(any());
    }

    @Test
    @DisplayName("keeps the not-found behaviour for an unknown category")
    void keepsNotFoundBehaviourForUnknownCategory() {
        // Given
        when(faqCategoryRepository.findById(404L)).thenReturn(Optional.empty());

        // When / Then
        assertThatThrownBy(() -> service.reorderCategory(404L, List.of(1L)))
                .isInstanceOf(ResourceNotFoundException.class);
        verify(faqRepository, never()).save(any());
    }
}
