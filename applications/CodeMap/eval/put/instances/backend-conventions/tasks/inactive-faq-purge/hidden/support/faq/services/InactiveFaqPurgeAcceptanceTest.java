package com.sm.instagram.platform.support.faq.services;

import com.sm.instagram.platform.support.faq.models.Faq;
import com.sm.instagram.platform.support.faq.repositories.FaqCategoryRepository;
import com.sm.instagram.platform.support.faq.repositories.FaqRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.invocation.Invocation;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Hidden acceptance test of the task inactive-faq-purge (prompt under test, instance backend-conventions). */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class InactiveFaqPurgeAcceptanceTest {

    @Mock FaqService faqServiceForJob;
    @InjectMocks InactiveFaqPurgeCronJob job;

    @Mock FaqRepository faqRepository;
    @Mock FaqCategoryRepository faqCategoryRepository;
    @InjectMocks FaqService service;

    @Test
    void theJobPurgesWithTheConfiguredAge() {
        ReflectionTestUtils.setField(job, "enabled", true);
        ReflectionTestUtils.setField(job, "olderThanDays", 90);
        job.purgeInactiveFaqs();
        verify(faqServiceForJob).purgeInactiveOlderThan(90);
    }

    @Test
    void aDisabledJobDoesNothing() {
        ReflectionTestUtils.setField(job, "enabled", false);
        ReflectionTestUtils.setField(job, "olderThanDays", 90);
        job.purgeInactiveFaqs();
        verify(faqServiceForJob, never()).purgeInactiveOlderThan(anyInt());
    }

    @Test
    void aFailureIsNotThrownOutOfTheJob() {
        ReflectionTestUtils.setField(job, "enabled", true);
        ReflectionTestUtils.setField(job, "olderThanDays", 90);
        when(faqServiceForJob.purgeInactiveOlderThan(anyInt())).thenThrow(new IllegalStateException("db down"));
        assertThatCode(() -> job.purgeInactiveFaqs()).doesNotThrowAnyException();
    }

    @Test
    void theServiceDeletesTheInactiveFaqsBeforeTheCutoff() {
        Faq first = new Faq();
        first.setId(1L);
        Faq second = new Faq();
        second.setId(2L);
        when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class)))
                .thenReturn(new ArrayList<>(List.of(first, second)));

        int deleted = service.purgeInactiveOlderThan(180);

        assertThat(deleted).isEqualTo(2);
        ArgumentCaptor<LocalDateTime> cutoff = ArgumentCaptor.forClass(LocalDateTime.class);
        verify(faqRepository).findByActiveFalseAndLastUpdateTimeBefore(cutoff.capture());
        assertThat(Duration.between(cutoff.getValue(), LocalDateTime.now().minusDays(180)).abs()).isLessThan(Duration.ofMinutes(5));
        assertThat(deletedIds(faqRepository)).contains(1L, 2L);
    }

    /** The ids of whatever the service handed to delete, deleteAll, deleteAllInBatch, deleteById or deleteAllById. */
    private static List<Long> deletedIds(Object repository) {
        List<Object> args = new ArrayList<>();
        for (Invocation i : Mockito.mockingDetails(repository).getInvocations()) {
            if (!i.getMethod().getName().startsWith("delete")) continue;
            for (Object arg : i.getArguments()) {
                if (arg instanceof Collection<?> c) args.addAll(c);
                else if (arg instanceof Iterable<?> it) it.forEach(args::add);
                else args.add(arg);
            }
        }
        List<Long> ids = new ArrayList<>();
        for (Object o : args) {
            if (o instanceof Faq f) ids.add(f.getId());
            else if (o instanceof Long id) ids.add(id);
        }
        return ids;
    }
}
