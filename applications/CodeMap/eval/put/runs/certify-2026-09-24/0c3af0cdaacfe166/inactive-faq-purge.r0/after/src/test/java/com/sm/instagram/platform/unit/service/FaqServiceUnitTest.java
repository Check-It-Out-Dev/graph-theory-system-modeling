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
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.modelmapper.ModelMapper;
import org.springframework.context.ApplicationContext;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

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
                applicationContext
        );
    }

    @Nested
    @DisplayName("purgeInactiveOlderThan")
    class PurgeInactiveOlderThan {

        @Test
        @DisplayName("should delete inactive FAQs older than the cutoff and return the count")
        void should_delete_inactive_faqs_and_return_count() {
            Faq faq1 = new Faq();
            faq1.setId(1L);
            Faq faq2 = new Faq();
            faq2.setId(2L);
            List<Faq> inactiveFaqs = List.of(faq1, faq2);

            when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                    .thenReturn(inactiveFaqs);

            int deleted = service.purgeInactiveOlderThan(180);

            assertThat(deleted).isEqualTo(2);
            verify(faqRepository).deleteAll(inactiveFaqs);
        }

        @Test
        @DisplayName("should return zero and not delete when there is nothing to purge")
        void should_return_zero_when_nothing_to_purge() {
            when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                    .thenReturn(List.of());

            int deleted = service.purgeInactiveOlderThan(180);

            assertThat(deleted).isEqualTo(0);
            verify(faqRepository).deleteAll(List.of());
        }

        @Test
        @DisplayName("should compute the cutoff from the given number of days")
        void should_compute_cutoff_from_days() {
            when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                    .thenReturn(List.of());

            LocalDateTime before = LocalDateTime.now().minusDays(30);
            service.purgeInactiveOlderThan(30);
            LocalDateTime after = LocalDateTime.now().minusDays(30);

            ArgumentCaptor<LocalDateTime> captor = ArgumentCaptor.forClass(LocalDateTime.class);
            verify(faqRepository).findByActiveFalseAndLastUpdateTimeBefore(captor.capture());
            assertThat(captor.getValue()).isBetween(before, after);
        }
    }
}
