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
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
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

    private Faq createFaq(Long id) {
        Faq faq = new Faq();
        faq.setId(id);
        faq.setQuestion("Question " + id);
        faq.setAnswer("Answer " + id);
        faq.setActive(false);
        return faq;
    }

    @Nested
    @DisplayName("purgeInactiveOlderThan")
    class PurgeInactiveOlderThan {

        @Test
        @DisplayName("should delete every inactive FAQ older than the cutoff and return the count")
        void should_delete_inactive_faqs_older_than_cutoff() {
            Faq faq1 = createFaq(11L);
            Faq faq2 = createFaq(22L);
            Faq faq3 = createFaq(33L);
            List<Faq> stale = List.of(faq1, faq2, faq3);
            when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                    .thenReturn(stale);

            int purged = service.purgeInactiveOlderThan(30);

            assertThat(purged).isEqualTo(3);
            verify(faqRepository).deleteAll(stale);
        }

        @Test
        @DisplayName("should pass a cutoff of now minus the given days")
        void should_use_cutoff_of_now_minus_days() {
            when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                    .thenReturn(Collections.emptyList());
            LocalDateTime before = LocalDateTime.now().minusDays(180);

            service.purgeInactiveOlderThan(180);

            LocalDateTime after = LocalDateTime.now().minusDays(180);
            ArgumentCaptor<LocalDateTime> cutoffCaptor = ArgumentCaptor.forClass(LocalDateTime.class);
            verify(faqRepository).findByActiveFalseAndLastUpdateTimeBefore(cutoffCaptor.capture());
            LocalDateTime cutoff = cutoffCaptor.getValue();
            assertThat(cutoff).isBetween(before, after);
        }

        @Test
        @DisplayName("should return zero and delete nothing when no FAQ qualifies")
        void should_return_zero_when_no_faq_qualifies() {
            when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                    .thenReturn(Collections.emptyList());

            int purged = service.purgeInactiveOlderThan(180);

            assertThat(purged).isZero();
            verify(faqRepository).deleteAll(Collections.emptyList());
        }
    }
}
