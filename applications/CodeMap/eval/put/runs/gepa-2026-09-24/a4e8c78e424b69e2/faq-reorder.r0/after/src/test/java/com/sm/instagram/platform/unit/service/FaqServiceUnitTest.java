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
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.modelmapper.ModelMapper;
import org.springframework.context.ApplicationContext;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for FaqService#reorderCategory.
 */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
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

    @Mock
    private SecurityContext securityContext;

    @Mock
    private Authentication authentication;

    private FaqService service;

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

        when(applicationContext.getBean(FaqService.class)).thenReturn(service);

        SecurityContextHolder.setContext(securityContext);
        when(securityContext.getAuthentication()).thenReturn(authentication);
        when(authentication.getName()).thenReturn("admin-user-id");
    }

    private FaqCategory createCategory(Long id) {
        FaqCategory category = new FaqCategory();
        category.setId(id);
        category.setName("Account");
        return category;
    }

    private Faq createFaq(Long id, FaqCategory category, int displayOrder) {
        Faq faq = new Faq();
        faq.setId(id);
        faq.setCategory(category);
        faq.setQuestion("Question " + id);
        faq.setAnswer("Answer " + id);
        faq.setDisplayOrder(displayOrder);
        faq.setActive(true);
        return faq;
    }

    @Nested
    @DisplayName("reorderCategory")
    class ReorderCategoryTests {

        @Test
        @DisplayName("should assign display order 1..n following the submitted order and save each FAQ")
        void shouldReorderFaqsAccordingToSubmittedOrder() {
            // Given
            FaqCategory category = createCategory(1L);
            Faq faq1 = createFaq(1L, category, 1);
            Faq faq2 = createFaq(2L, category, 2);
            Faq faq3 = createFaq(3L, category, 3);

            when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Arrays.asList(faq1, faq2, faq3));
            when(faqRepository.save(any(Faq.class))).thenAnswer(inv -> inv.getArgument(0));

            // When
            service.reorderCategory(1L, Arrays.asList(3L, 1L, 2L));

            // Then
            assertThat(faq3.getDisplayOrder()).isEqualTo(1);
            assertThat(faq1.getDisplayOrder()).isEqualTo(2);
            assertThat(faq2.getDisplayOrder()).isEqualTo(3);
            verify(faqRepository).save(faq1);
            verify(faqRepository).save(faq2);
            verify(faqRepository).save(faq3);
        }

        @Test
        @DisplayName("should reject a list missing one of the category's active FAQs")
        void shouldRejectListMissingAFaq() {
            // Given
            FaqCategory category = createCategory(1L);
            Faq faq1 = createFaq(1L, category, 1);
            Faq faq2 = createFaq(2L, category, 2);

            when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Arrays.asList(faq1, faq2));

            // When/Then
            assertThatThrownBy(() -> service.reorderCategory(1L, List.of(1L)))
                    .isInstanceOf(BusinessRuleTranslatableException.class)
                    .hasFieldOrPropertyWithValue("messageKey", "error.business.faq_reorder_mismatch");
        }

        @Test
        @DisplayName("should reject a list with a duplicated id in place of another FAQ")
        void shouldRejectListWithDuplicateId() {
            // Given
            FaqCategory category = createCategory(1L);
            Faq faq1 = createFaq(1L, category, 1);
            Faq faq2 = createFaq(2L, category, 2);

            when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Arrays.asList(faq1, faq2));

            // When/Then
            assertThatThrownBy(() -> service.reorderCategory(1L, Arrays.asList(1L, 1L)))
                    .isInstanceOf(BusinessRuleTranslatableException.class)
                    .hasFieldOrPropertyWithValue("messageKey", "error.business.faq_reorder_mismatch");
        }

        @Test
        @DisplayName("should reject an id that does not belong to the category")
        void shouldRejectIdFromAnotherCategory() {
            // Given
            FaqCategory category = createCategory(1L);
            Faq faq1 = createFaq(1L, category, 1);
            Faq faq2 = createFaq(2L, category, 2);

            when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Arrays.asList(faq1, faq2));

            // When/Then
            assertThatThrownBy(() -> service.reorderCategory(1L, Arrays.asList(1L, 99L)))
                    .isInstanceOf(BusinessRuleTranslatableException.class)
                    .hasFieldOrPropertyWithValue("messageKey", "error.business.faq_reorder_mismatch");
        }

        @Test
        @DisplayName("should keep the existing not-found behaviour for an unknown category")
        void shouldKeepNotFoundBehaviourForUnknownCategory() {
            // Given
            when(faqCategoryRepository.findById(999L)).thenReturn(Optional.empty());

            // When/Then
            assertThatThrownBy(() -> service.reorderCategory(999L, List.of(1L)))
                    .isInstanceOf(ResourceNotFoundException.class)
                    .hasFieldOrPropertyWithValue("messageKey", "error.business.item_not_found");
        }
    }
}
