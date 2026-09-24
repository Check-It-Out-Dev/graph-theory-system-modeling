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
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@DisplayName("FaqService Purge Unit Tests")
class FaqService_PurgeUnitTest {

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
        @DisplayName("should delete inactive FAQs older than the cutoff and return their count")
        void should_delete_inactive_faqs_older_than_cutoff() {
            Faq first = new Faq();
            first.setId(1L);
            Faq second = new Faq();
            second.setId(2L);
            Faq third = new Faq();
            third.setId(3L);
            List<Faq> staleFaqs = List.of(first, second, third);

            when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                    .thenReturn(staleFaqs);

            LocalDateTime before = LocalDateTime.now().minusDays(90);
            int deleted = faqService.purgeInactiveOlderThan(90);
            LocalDateTime after = LocalDateTime.now().minusDays(90);

            assertThat(deleted).isEqualTo(3);

            ArgumentCaptor<LocalDateTime> cutoffCaptor = ArgumentCaptor.forClass(LocalDateTime.class);
            verify(faqRepository).findByActiveFalseAndLastUpdateTimeBefore(cutoffCaptor.capture());
            LocalDateTime cutoff = cutoffCaptor.getValue();
            assertThat(cutoff).isBetween(before.minus(1, ChronoUnit.SECONDS), after.plus(1, ChronoUnit.SECONDS));

            verify(faqRepository).deleteAll(staleFaqs);
        }

        @Test
        @DisplayName("should return zero and delete nothing when no FAQ is inactive long enough")
        void should_return_zero_when_no_faq_is_stale_enough() {
            when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                    .thenReturn(Collections.emptyList());

            int deleted = faqService.purgeInactiveOlderThan(180);

            assertThat(deleted).isEqualTo(0);
            verify(faqRepository).deleteAll(Collections.emptyList());
        }
    }
}
