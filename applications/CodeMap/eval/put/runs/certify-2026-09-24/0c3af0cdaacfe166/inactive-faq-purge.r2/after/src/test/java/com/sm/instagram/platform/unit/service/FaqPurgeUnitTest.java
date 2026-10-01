package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.util.RepositoryResolver;
import com.sm.instagram.platform.common.util.filtering.SpecificationBuilder;
import com.sm.instagram.platform.support.faq.models.Faq;
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
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.modelmapper.ModelMapper;
import org.springframework.context.ApplicationContext;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
@DisplayName("FaqService.purgeInactiveOlderThan Unit Tests")
class FaqPurgeUnitTest {

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

    private FaqService faqService;

    @BeforeEach
    void setUp() {
        faqService = new FaqService(
                specificationBuilder,
                faqRepository,
                faqCategoryRepository,
                modelMapper,
                repositoryResolver,
                applicationContext);
    }

    @Test
    @DisplayName("should delete inactive FAQs older than the cutoff and return the deleted count")
    void should_delete_and_return_count() {
        Faq faq1 = new Faq();
        faq1.setId(1L);
        Faq faq2 = new Faq();
        faq2.setId(2L);
        when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                .thenReturn(List.of(faq1, faq2));

        int deleted = faqService.purgeInactiveOlderThan(180);

        assertThat(deleted).isEqualTo(2);
        verify(faqRepository).deleteAll(List.of(faq1, faq2));
    }

    @Test
    @DisplayName("should use a cutoff of now minus the given number of days")
    void should_use_cutoff_of_now_minus_days() {
        when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                .thenReturn(List.of());

        LocalDateTime before = LocalDateTime.now().minusDays(180);
        faqService.purgeInactiveOlderThan(180);
        LocalDateTime after = LocalDateTime.now().minusDays(180);

        ArgumentCaptor<LocalDateTime> cutoffCaptor = ArgumentCaptor.forClass(LocalDateTime.class);
        verify(faqRepository).findByActiveFalseAndLastUpdateTimeBefore(cutoffCaptor.capture());
        assertThat(cutoffCaptor.getValue()).isBetween(before, after);
    }

    @Test
    @DisplayName("should return zero and skip deletion when nothing is inactive long enough")
    void should_return_zero_when_nothing_to_purge() {
        when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                .thenReturn(List.of());

        int deleted = faqService.purgeInactiveOlderThan(180);

        assertThat(deleted).isZero();
        verify(faqRepository, never()).deleteAll(anyList());
    }
}
