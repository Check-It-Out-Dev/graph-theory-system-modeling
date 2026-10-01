package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.faq.models.Faq;
import com.sm.instagram.platform.support.faq.repositories.FaqRepository;
import com.sm.instagram.platform.support.faq.services.FaqService;
import com.sm.instagram.platform.support.faq.services.InactiveFaqPurgeCronJob;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class InactiveFaqPurgeUnitTest {

    @Mock
    private FaqRepository faqRepository;

    @InjectMocks
    private FaqService faqService;

    @Test
    void deletesTheInactiveFaqsBeforeTheCutoff() {
        List<Faq> stale = List.of(new Faq(), new Faq());
        when(faqRepository.findByActiveFalseAndLastUpdateTimeBefore(any(LocalDateTime.class))).thenReturn(stale);

        assertThat(faqService.purgeInactiveOlderThan(180)).isEqualTo(2);
        verify(faqRepository).deleteAll(stale);
    }

    @Test
    void jobDelegatesWhenEnabledAndSkipsWhenDisabled() {
        FaqService service = mock(FaqService.class);
        InactiveFaqPurgeCronJob job = new InactiveFaqPurgeCronJob(service);
        ReflectionTestUtils.setField(job, "enabled", true);
        ReflectionTestUtils.setField(job, "olderThanDays", 180);
        job.purgeInactiveFaqs();
        verify(service).purgeInactiveOlderThan(180);

        FaqService idle = mock(FaqService.class);
        InactiveFaqPurgeCronJob disabled = new InactiveFaqPurgeCronJob(idle);
        ReflectionTestUtils.setField(disabled, "enabled", false);
        disabled.purgeInactiveFaqs();
        verify(idle, never()).purgeInactiveOlderThan(anyInt());
    }

    @Test
    void jobSwallowsFailures() {
        FaqService service = mock(FaqService.class);
        when(service.purgeInactiveOlderThan(anyInt())).thenThrow(new IllegalStateException("db"));
        InactiveFaqPurgeCronJob job = new InactiveFaqPurgeCronJob(service);
        ReflectionTestUtils.setField(job, "enabled", true);
        assertThatCode(job::purgeInactiveFaqs).doesNotThrowAnyException();
    }
}
