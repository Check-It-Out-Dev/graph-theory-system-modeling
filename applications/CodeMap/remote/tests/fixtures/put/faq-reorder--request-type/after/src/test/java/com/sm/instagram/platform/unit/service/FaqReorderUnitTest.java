package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.support.faq.models.Faq;
import com.sm.instagram.platform.support.faq.models.FaqCategory;
import com.sm.instagram.platform.support.faq.repositories.FaqCategoryRepository;
import com.sm.instagram.platform.support.faq.repositories.FaqRepository;
import com.sm.instagram.platform.support.faq.services.FaqService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyIterable;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class FaqReorderUnitTest {

    @Mock
    private FaqRepository faqRepository;

    @Mock
    private FaqCategoryRepository faqCategoryRepository;

    @InjectMocks
    private FaqService faqService;

    @Test
    void assignsDisplayOrderFromThePosition() {
        FaqCategory category = category();
        Faq first = faq(1L);
        Faq second = faq(2L);
        when(faqCategoryRepository.findById(7L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category)).thenReturn(List.of(first, second));

        faqService.reorderCategory(7L, List.of(2L, 1L));

        assertThat(second.getDisplayOrder()).isEqualTo(1);
        assertThat(first.getDisplayOrder()).isEqualTo(2);
        verify(faqRepository).saveAll(List.of(second, first));
    }

    @Test
    void refusesAListThatDoesNotMatchTheActiveFaqs() {
        FaqCategory category = category();
        when(faqCategoryRepository.findById(7L)).thenReturn(Optional.of(category));
        when(faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category)).thenReturn(List.of(faq(1L), faq(2L)));

        assertThatThrownBy(() -> faqService.reorderCategory(7L, List.of(1L)))
                .isInstanceOf(BusinessRuleTranslatableException.class)
                .hasMessage("error.business.faq_reorder_mismatch");
        verify(faqRepository, never()).saveAll(anyIterable());
    }

    private static FaqCategory category() {
        FaqCategory c = new FaqCategory();
        c.setId(7L);
        return c;
    }

    private static Faq faq(Long id) {
        Faq f = new Faq();
        f.setId(id);
        f.setActive(true);
        return f;
    }
}
