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
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
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
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link FaqService#reorderCategory(Long, List)}.
 */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
@DisplayName("FaqService reorderCategory Unit Tests")
class FaqServiceReorderUnitTest {

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
        when(authentication.getName()).thenReturn("admin-uid");
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

    @Test
    @DisplayName("should assign display order 1..n following the given order")
    void shouldReorderActiveFaqsInGivenOrder() {
        FaqCategory category = new FaqCategory();
        category.setId(1L);

        Faq faq10 = createFaq(10L, category, 1);
        Faq faq20 = createFaq(20L, category, 2);
        Faq faq30 = createFaq(30L, category, 3);

        when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq10, faq20, faq30));
        when(faqRepository.save(any(Faq.class))).thenAnswer(inv -> inv.getArgument(0));

        service.reorderCategory(1L, List.of(30L, 10L, 20L));

        assertThat(faq30.getDisplayOrder()).isEqualTo(1);
        assertThat(faq10.getDisplayOrder()).isEqualTo(2);
        assertThat(faq20.getDisplayOrder()).isEqualTo(3);
        verify(faqRepository, org.mockito.Mockito.times(3)).save(any(Faq.class));
    }

    @Test
    @DisplayName("should reject a list missing one of the category's active FAQs")
    void shouldRejectMissingFaq() {
        FaqCategory category = new FaqCategory();
        category.setId(1L);

        Faq faq10 = createFaq(10L, category, 1);
        Faq faq20 = createFaq(20L, category, 2);

        when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq10, faq20));

        assertThatThrownBy(() -> service.reorderCategory(1L, List.of(10L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> assertThat(((BusinessRuleTranslatableException) ex).getMessageKey())
                        .isEqualTo("error.business.faq_reorder_mismatch"));
    }

    @Test
    @DisplayName("should reject a list with an id from outside the category")
    void shouldRejectForeignFaqId() {
        FaqCategory category = new FaqCategory();
        category.setId(1L);

        Faq faq10 = createFaq(10L, category, 1);

        when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq10));

        assertThatThrownBy(() -> service.reorderCategory(1L, List.of(10L, 999L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> assertThat(((BusinessRuleTranslatableException) ex).getMessageKey())
                        .isEqualTo("error.business.faq_reorder_mismatch"));
    }

    @Test
    @DisplayName("should reject a list with a duplicated id")
    void shouldRejectDuplicatedFaqId() {
        FaqCategory category = new FaqCategory();
        category.setId(1L);

        Faq faq10 = createFaq(10L, category, 1);
        Faq faq20 = createFaq(20L, category, 2);

        when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category))
                .thenReturn(List.of(faq10, faq20));

        assertThatThrownBy(() -> service.reorderCategory(1L, List.of(10L, 10L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .satisfies(ex -> assertThat(((BusinessRuleTranslatableException) ex).getMessageKey())
                        .isEqualTo("error.business.faq_reorder_mismatch"));
    }

    @Test
    @DisplayName("should keep the existing not-found behaviour for an unknown category")
    void shouldThrowWhenCategoryNotFound() {
        when(faqCategoryRepository.findById(99L)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.reorderCategory(99L, List.of(1L)))
                .isInstanceOf(ResourceNotFoundException.class);
    }
}
