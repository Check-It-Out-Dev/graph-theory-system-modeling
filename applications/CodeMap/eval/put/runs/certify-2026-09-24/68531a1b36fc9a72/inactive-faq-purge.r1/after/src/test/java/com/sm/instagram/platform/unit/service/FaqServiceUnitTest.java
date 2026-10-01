package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.util.RepositoryResolver;
import com.sm.instagram.platform.common.util.filtering.SpecificationBuilder;
import com.sm.instagram.platform.support.faq.models.Faq;
import com.sm.instagram.platform.support.faq.repositories.FaqCategoryRepository;
import com.sm.instagram.platform.support.faq.repositories.FaqRepository;
import com.sm.instagram.platform.support.faq.services.FaqService;
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

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

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

    @Nested
    @DisplayName("purgeInactiveOlderThan")
    class PurgeInactiveOlderThan {

        @Test
        @DisplayName("should delete every inactive FAQ found before the cutoff and return the deleted count")
        void should_delete_stale_inactive_faqs_and_return_count() {
            int retentionDays = 200;
            Faq stale1 = new Faq();
            stale1.setId(11L);
            Faq stale2 = new Faq();
            stale2.setId(22L);
            List<Faq> staleFaqs = List.of(stale1, stale2);

            when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any())).thenReturn(staleFaqs);

            int result = faqService.purgeInactiveOlderThan(retentionDays);

            ArgumentCaptor<LocalDateTime> cutoffCaptor = ArgumentCaptor.forClass(LocalDateTime.class);
            verify(faqRepository).findByActiveFalseAndLastUpdateTimeBefore(cutoffCaptor.capture());
            LocalDateTime expectedCutoff = LocalDateTime.now().minusDays(retentionDays);
            assertThat(cutoffCaptor.getValue()).isCloseTo(expectedCutoff, within(5, ChronoUnit.SECONDS));

            verify(faqRepository).deleteAll(staleFaqs);
            assertThat(result).isEqualTo(2);
        }

        @Test
        @DisplayName("should return zero and delete nothing when no inactive FAQ is old enough")
        void should_return_zero_when_no_stale_faqs_found() {
            when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any())).thenReturn(List.of());

            int result = faqService.purgeInactiveOlderThan(180);

            verify(faqRepository).deleteAll(List.of());
            assertThat(result).isZero();
        }
    }
}
