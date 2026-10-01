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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.modelmapper.ModelMapper;
import org.springframework.context.ApplicationContext;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for FaqService.
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

    private Faq faq(Long id, int displayOrder) {
        Faq faq = new Faq();
        faq.setId(id);
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
        @DisplayName("should assign sequential display orders matching the requested order")
        void shouldAssignSequentialDisplayOrders() {
            // Given
            Long categoryId = 7L;
            FaqCategory category = new FaqCategory();
            category.setId(categoryId);

            Faq faq10 = faq(10L, 5);
            Faq faq20 = faq(20L, 1);
            Faq faq30 = faq(30L, 3);

            when(faqCategoryRepository.findById(categoryId)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Arrays.asList(faq20, faq30, faq10));
            when(faqRepository.save(any(Faq.class))).thenAnswer(invocation -> invocation.getArgument(0));

            SecurityContext context = SecurityContextHolder.createEmptyContext();
            context.setAuthentication(new UsernamePasswordAuthenticationToken("admin-1", null, Collections.emptyList()));
            SecurityContextHolder.setContext(context);
            when(applicationContext.getBean(FaqService.class)).thenReturn(service);

            // When
            service.reorderCategory(categoryId, Arrays.asList(20L, 30L, 10L));

            // Then
            assertThat(faq20.getDisplayOrder()).isEqualTo(1);
            assertThat(faq30.getDisplayOrder()).isEqualTo(2);
            assertThat(faq10.getDisplayOrder()).isEqualTo(3);
            verify(faqRepository).save(faq20);
            verify(faqRepository).save(faq30);
            verify(faqRepository).save(faq10);
        }

        @Test
        @DisplayName("should do nothing when the category has no active FAQs and the list is empty")
        void shouldHandleEmptyCategory() {
            // Given
            Long categoryId = 8L;
            FaqCategory category = new FaqCategory();
            category.setId(categoryId);

            when(faqCategoryRepository.findById(categoryId)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Collections.emptyList());

            // When / Then
            org.junit.jupiter.api.Assertions.assertDoesNotThrow(
                    () -> service.reorderCategory(categoryId, Collections.emptyList()));
            verify(faqRepository, never()).save(any(Faq.class));
        }

        @Test
        @DisplayName("should throw faq_reorder_mismatch when an id is missing")
        void shouldThrowWhenIdIsMissing() {
            // Given
            Long categoryId = 9L;
            FaqCategory category = new FaqCategory();
            category.setId(categoryId);

            Faq faq1 = faq(1L, 1);
            Faq faq2 = faq(2L, 2);

            when(faqCategoryRepository.findById(categoryId)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Arrays.asList(faq1, faq2));

            // When / Then
            assertThatThrownBy(() -> service.reorderCategory(categoryId, List.of(1L)))
                    .isInstanceOf(BusinessRuleTranslatableException.class)
                    .satisfies(ex -> assertThat(((BusinessRuleTranslatableException) ex).getMessageKey())
                            .isEqualTo("error.business.faq_reorder_mismatch"));
            verify(faqRepository, never()).save(any(Faq.class));
        }

        @Test
        @DisplayName("should throw faq_reorder_mismatch when an id is duplicated")
        void shouldThrowWhenIdIsDuplicated() {
            // Given
            Long categoryId = 11L;
            FaqCategory category = new FaqCategory();
            category.setId(categoryId);

            Faq faq1 = faq(1L, 1);
            Faq faq2 = faq(2L, 2);

            when(faqCategoryRepository.findById(categoryId)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Arrays.asList(faq1, faq2));

            // When / Then
            assertThatThrownBy(() -> service.reorderCategory(categoryId, Arrays.asList(1L, 1L)))
                    .isInstanceOf(BusinessRuleTranslatableException.class)
                    .satisfies(ex -> assertThat(((BusinessRuleTranslatableException) ex).getMessageKey())
                            .isEqualTo("error.business.faq_reorder_mismatch"));
            verify(faqRepository, never()).save(any(Faq.class));
        }

        @Test
        @DisplayName("should throw faq_reorder_mismatch when an id belongs to another category")
        void shouldThrowWhenIdBelongsToAnotherCategory() {
            // Given
            Long categoryId = 12L;
            FaqCategory category = new FaqCategory();
            category.setId(categoryId);

            Faq faq1 = faq(1L, 1);
            Faq faq2 = faq(2L, 2);

            when(faqCategoryRepository.findById(categoryId)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Arrays.asList(faq1, faq2));

            // When / Then
            assertThatThrownBy(() -> service.reorderCategory(categoryId, Arrays.asList(1L, 99L)))
                    .isInstanceOf(BusinessRuleTranslatableException.class)
                    .satisfies(ex -> assertThat(((BusinessRuleTranslatableException) ex).getMessageKey())
                            .isEqualTo("error.business.faq_reorder_mismatch"));
            verify(faqRepository, never()).save(any(Faq.class));
        }

        @Test
        @DisplayName("should throw not-found for an unknown category")
        void shouldThrowNotFoundForUnknownCategory() {
            // Given
            Long categoryId = 404L;
            when(faqCategoryRepository.findById(categoryId)).thenReturn(Optional.empty());

            // When / Then
            assertThatThrownBy(() -> service.reorderCategory(categoryId, List.of(1L)))
                    .isInstanceOf(ResourceNotFoundException.class);
            verify(faqRepository, never()).findByCategoryAndActiveTrueOrderByDisplayOrderAsc(any(FaqCategory.class));
        }
    }
}
