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
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
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
    private FaqCategory category;
    private Faq faq1;
    private Faq faq2;
    private Faq faq3;

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

        // Setup getSelf() to return the service itself for transactional proxy simulation
        when(applicationContext.getBean(FaqService.class)).thenReturn(service);

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

        when(faqCategoryRepository.findById(1L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrue(category)).thenReturn(List.of(faq1, faq2, faq3));
        when(faqRepository.save(any(Faq.class))).thenAnswer(invocation -> invocation.getArgument(0));

        SecurityContext context = SecurityContextHolder.createEmptyContext();
        context.setAuthentication(new UsernamePasswordAuthenticationToken("admin-uid", null, List.of()));
        SecurityContextHolder.setContext(context);
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    @Test
    @DisplayName("assigns display order 1..n following the submitted order")
    void assignsDisplayOrderFollowingSubmittedOrder() {
        service.reorderCategory(1L, List.of(30L, 10L, 20L));

        assertThat(faq3.getDisplayOrder()).isEqualTo(1);
        assertThat(faq1.getDisplayOrder()).isEqualTo(2);
        assertThat(faq2.getDisplayOrder()).isEqualTo(3);
    }

    @Test
    @DisplayName("accepts the ids in any order as long as each active FAQ appears once")
    void acceptsIdsInAnyOrder() {
        service.reorderCategory(1L, List.of(20L, 30L, 10L));

        assertThat(faq2.getDisplayOrder()).isEqualTo(1);
        assertThat(faq3.getDisplayOrder()).isEqualTo(2);
        assertThat(faq1.getDisplayOrder()).isEqualTo(3);
    }

    @Test
    @DisplayName("rejects a list missing one of the active FAQs")
    void rejectsMissingFaq() {
        assertThatThrownBy(() -> service.reorderCategory(1L, List.of(10L, 20L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .hasFieldOrPropertyWithValue("messageKey", "error.business.faq_reorder_mismatch");
    }

    @Test
    @DisplayName("rejects a list with an id from another category")
    void rejectsUnknownFaq() {
        assertThatThrownBy(() -> service.reorderCategory(1L, List.of(10L, 20L, 999L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .hasFieldOrPropertyWithValue("messageKey", "error.business.faq_reorder_mismatch");
    }

    @Test
    @DisplayName("rejects a list with a duplicate id")
    void rejectsDuplicateFaq() {
        assertThatThrownBy(() -> service.reorderCategory(1L, List.of(10L, 10L, 20L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .hasFieldOrPropertyWithValue("messageKey", "error.business.faq_reorder_mismatch");
    }

    @Test
    @DisplayName("rejects a null list")
    void rejectsNullList() {
        assertThatThrownBy(() -> service.reorderCategory(1L, null))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .hasFieldOrPropertyWithValue("messageKey", "error.business.faq_reorder_mismatch");
    }

    @Test
    @DisplayName("keeps the existing not-found behaviour for an unknown category")
    void unknownCategoryThrowsNotFound() {
        when(faqCategoryRepository.findById(99L)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.reorderCategory(99L, List.of(10L)))
                .isInstanceOf(ResourceNotFoundException.class);
    }
}
