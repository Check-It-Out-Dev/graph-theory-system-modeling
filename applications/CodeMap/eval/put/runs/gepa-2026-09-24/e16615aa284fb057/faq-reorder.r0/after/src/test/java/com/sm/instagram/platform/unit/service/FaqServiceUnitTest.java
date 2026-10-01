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

/**
 * Unit tests for {@link FaqService#reorderCategory(Long, List)}.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("FaqService Unit Tests")
class FaqServiceUnitTest {

    private static final String PRINCIPAL = "admin-firebase-uid";

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
                applicationContext
        );
    }

    private FaqCategory category(Long id) {
        FaqCategory category = new FaqCategory();
        category.setId(id);
        category.setName("Account");
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

    @Nested
    @DisplayName("reorderCategory - unknown category")
    class UnknownCategoryTests {

        @Test
        @DisplayName("keeps the existing not-found behaviour")
        void throwsResourceNotFoundForUnknownCategory() {
            when(faqCategoryRepository.findById(404L)).thenReturn(Optional.empty());

            assertThatThrownBy(() -> service.reorderCategory(404L, List.of(1L)))
                    .isInstanceOf(ResourceNotFoundException.class);

            verify(faqRepository, never()).save(any(Faq.class));
        }
    }

    @Nested
    @DisplayName("reorderCategory - mismatched ids")
    class MismatchTests {

        @Test
        @DisplayName("rejects a list missing one of the category's active FAQs")
        void rejectsMissingId() {
            FaqCategory cat = category(7L);
            Faq faq1 = faq(10L, cat, 1);
            Faq faq2 = faq(11L, cat, 2);
            when(faqCategoryRepository.findById(7L)).thenReturn(Optional.of(cat));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(cat))
                    .thenReturn(List.of(faq1, faq2));

            assertThatThrownBy(() -> service.reorderCategory(7L, List.of(10L)))
                    .isInstanceOf(BusinessRuleTranslatableException.class)
                    .satisfies(e -> assertThat(((BusinessRuleTranslatableException) e).getMessageKey())
                            .isEqualTo("error.business.faq_reorder_mismatch"));

            verify(faqRepository, never()).save(any(Faq.class));
        }

        @Test
        @DisplayName("rejects a list with a duplicated id even when the count matches")
        void rejectsDuplicateId() {
            FaqCategory cat = category(7L);
            Faq faq1 = faq(10L, cat, 1);
            Faq faq2 = faq(11L, cat, 2);
            when(faqCategoryRepository.findById(7L)).thenReturn(Optional.of(cat));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(cat))
                    .thenReturn(List.of(faq1, faq2));

            assertThatThrownBy(() -> service.reorderCategory(7L, List.of(10L, 10L)))
                    .isInstanceOf(BusinessRuleTranslatableException.class);

            verify(faqRepository, never()).save(any(Faq.class));
        }

        @Test
        @DisplayName("rejects an id that belongs to another category")
        void rejectsIdFromAnotherCategory() {
            FaqCategory cat = category(7L);
            Faq faq1 = faq(10L, cat, 1);
            when(faqCategoryRepository.findById(7L)).thenReturn(Optional.of(cat));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(cat))
                    .thenReturn(List.of(faq1));

            assertThatThrownBy(() -> service.reorderCategory(7L, List.of(999L)))
                    .isInstanceOf(BusinessRuleTranslatableException.class);

            verify(faqRepository, never()).save(any(Faq.class));
        }
    }

    @Nested
    @DisplayName("reorderCategory - valid reorder")
    class ValidReorderTests {

        @BeforeEach
        void authenticate() {
            SecurityContext context = SecurityContextHolder.createEmptyContext();
            context.setAuthentication(new UsernamePasswordAuthenticationToken(PRINCIPAL, null, List.of()));
            SecurityContextHolder.setContext(context);
            when(applicationContext.getBean(FaqService.class)).thenReturn(service);
        }

        @AfterEach
        void forget() {
            SecurityContextHolder.clearContext();
        }

        @Test
        @DisplayName("assigns display order i + 1 to the FAQ at position i, in the requested order")
        void reordersFaqsByRequestedPositions() {
            FaqCategory cat = category(7L);
            Faq faq1 = faq(10L, cat, 1);
            Faq faq2 = faq(11L, cat, 2);
            Faq faq3 = faq(12L, cat, 3);
            when(faqCategoryRepository.findById(7L)).thenReturn(Optional.of(cat));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(cat))
                    .thenReturn(List.of(faq1, faq2, faq3));
            when(faqRepository.save(any(Faq.class))).thenAnswer(inv -> inv.getArgument(0));

            service.reorderCategory(7L, List.of(12L, 10L, 11L));

            assertThat(faq3.getDisplayOrder()).isEqualTo(1);
            assertThat(faq1.getDisplayOrder()).isEqualTo(2);
            assertThat(faq2.getDisplayOrder()).isEqualTo(3);

            ArgumentCaptor<Faq> captor = ArgumentCaptor.forClass(Faq.class);
            verify(faqRepository, times(3)).save(captor.capture());
            List<Faq> saved = captor.getAllValues();
            assertThat(saved).extracting(Faq::getId).containsExactly(12L, 10L, 11L);
            assertThat(saved).extracting(Faq::getUpdaterId).containsOnly(PRINCIPAL);
        }

        @Test
        @DisplayName("accepts a single-FAQ category")
        void reordersSingleFaqCategory() {
            FaqCategory cat = category(8L);
            Faq onlyFaq = faq(20L, cat, 1);
            when(faqCategoryRepository.findById(8L)).thenReturn(Optional.of(cat));
            when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(cat))
                    .thenReturn(List.of(onlyFaq));
            when(faqRepository.save(any(Faq.class))).thenAnswer(inv -> inv.getArgument(0));

            service.reorderCategory(8L, List.of(20L));

            assertThat(onlyFaq.getDisplayOrder()).isEqualTo(1);
            verify(faqRepository).save(onlyFaq);
        }
    }
}
