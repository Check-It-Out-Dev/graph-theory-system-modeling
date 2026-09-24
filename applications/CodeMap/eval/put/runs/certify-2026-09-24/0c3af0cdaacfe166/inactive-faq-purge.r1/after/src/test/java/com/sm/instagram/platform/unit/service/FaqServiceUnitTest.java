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
import static org.mockito.Mockito.*;

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
        @DisplayName("should delete inactive FAQs older than cutoff and return count")
        void should_delete_inactive_faqs_older_than_cutoff() {
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
        @DisplayName("should return zero when no inactive FAQs are old enough")
        void should_return_zero_when_nothing_to_purge() {
            when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                    .thenReturn(List.of());

            int deleted = service.purgeInactiveOlderThan(180);

            assertThat(deleted).isZero();
            verify(faqRepository).deleteAll(List.of());
        }

        @Test
        @DisplayName("should use a cutoff of now minus the given number of days")
        void should_use_cutoff_of_now_minus_days() {
            ArgumentCaptor<LocalDateTime> cutoffCaptor = ArgumentCaptor.forClass(LocalDateTime.class);
            when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(cutoffCaptor.capture()))
                    .thenReturn(List.of());

            LocalDateTime before = LocalDateTime.now().minusDays(180);
            service.purgeInactiveOlderThan(180);
            LocalDateTime after = LocalDateTime.now().minusDays(180);

            assertThat(cutoffCaptor.getValue()).isBetween(before, after);
        }
    }
}
