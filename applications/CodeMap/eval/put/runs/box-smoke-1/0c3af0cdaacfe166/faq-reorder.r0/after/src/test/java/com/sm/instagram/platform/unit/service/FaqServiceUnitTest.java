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
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link FaqService#reorderCategory(Long, List)}.
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

        when(applicationContext.getBean(FaqService.class)).thenReturn(service);

        SecurityContext securityContext = new org.springframework.security.core.context.SecurityContextImpl(
                new UsernamePasswordAuthenticationToken("admin-user", null, Collections.emptyList()));
        SecurityContextHolder.setContext(securityContext);
    }

    private Faq faq(Long id, int displayOrder) {
        Faq faq = new Faq();
        faq.setId(id);
        faq.setDisplayOrder(displayOrder);
        faq.setActive(true);
        return faq;
    }

    @Nested
    @DisplayName("reorderCategory")
    class ReorderCategoryTests {

        @Test
        @DisplayName("should assign display orders matching the given order")
        void shouldReorderFaqs() {
            FaqCategory category = new FaqCategory();
            category.setId(1L);

            Faq faq1 = faq(1L, 1);
            Faq faq2 = faq(2L, 2);
            Faq faq3 = faq(3L, 3);

            when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Arrays.asList(faq1, faq2, faq3));
            when(faqRepository.save(any(Faq.class))).thenAnswer(invocation -> invocation.getArgument(0));

            service.reorderCategory(1L, Arrays.asList(3L, 1L, 2L));

            assertThat(faq3.getDisplayOrder()).isEqualTo(1);
            assertThat(faq1.getDisplayOrder()).isEqualTo(2);
            assertThat(faq2.getDisplayOrder()).isEqualTo(3);
            verify(faqRepository, org.mockito.Mockito.times(3)).save(any(Faq.class));
        }

        @Test
        @DisplayName("should throw not found for an unknown category")
        void shouldThrowNotFoundForUnknownCategory() {
            when(faqCategoryRepository.findById(99L)).thenReturn(Optional.empty());

            assertThatThrownBy(() -> service.reorderCategory(99L, List.of(1L)))
                    .isInstanceOf(ResourceNotFoundException.class);
        }

        @Test
        @DisplayName("should throw a mismatch error when an id is missing")
        void shouldThrowMismatchWhenIdMissing() {
            FaqCategory category = new FaqCategory();
            category.setId(1L);

            Faq faq1 = faq(1L, 1);
            Faq faq2 = faq(2L, 2);

            when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Arrays.asList(faq1, faq2));

            assertThatThrownBy(() -> service.reorderCategory(1L, List.of(1L)))
                    .isInstanceOf(BusinessRuleTranslatableException.class)
                    .hasFieldOrPropertyWithValue("messageKey", "error.business.faq_reorder_mismatch");
        }

        @Test
        @DisplayName("should throw a mismatch error when an id does not belong to the category")
        void shouldThrowMismatchWhenIdUnknown() {
            FaqCategory category = new FaqCategory();
            category.setId(1L);

            Faq faq1 = faq(1L, 1);
            Faq faq2 = faq(2L, 2);

            when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Arrays.asList(faq1, faq2));

            assertThatThrownBy(() -> service.reorderCategory(1L, Arrays.asList(1L, 999L)))
                    .isInstanceOf(BusinessRuleTranslatableException.class)
                    .hasFieldOrPropertyWithValue("messageKey", "error.business.faq_reorder_mismatch");
        }

        @Test
        @DisplayName("should throw a mismatch error when an id is duplicated")
        void shouldThrowMismatchWhenIdDuplicated() {
            FaqCategory category = new FaqCategory();
            category.setId(1L);

            Faq faq1 = faq(1L, 1);
            Faq faq2 = faq(2L, 2);
            Faq faq3 = faq(3L, 3);

            when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                    .thenReturn(Arrays.asList(faq1, faq2, faq3));

            assertThatThrownBy(() -> service.reorderCategory(1L, Arrays.asList(1L, 1L, 2L)))
                    .isInstanceOf(BusinessRuleTranslatableException.class)
                    .hasFieldOrPropertyWithValue("messageKey", "error.business.faq_reorder_mismatch");
        }
    }
}
