package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.faq.services.FaqService;
import com.sm.instagram.platform.support.faq.services.InactiveFaqPurgeCronJob;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.util.ReflectionTestUtils;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
@DisplayName("InactiveFaqPurgeCronJob Unit Tests")
class InactiveFaqPurgeCronJobUnitTest {

    @Mock
    private FaqService faqService;

    private InactiveFaqPurgeCronJob cronJob;

    @BeforeEach
    void setUp() {
        cronJob = new InactiveFaqPurgeCronJob(faqService);
        ReflectionTestUtils.setField(cronJob, "olderThanDays", 180);
    }

    @Nested
    @DisplayName("purgeInactiveFaqs")
    class PurgeInactiveFaqs {

        @Test
        @DisplayName("should purge using the configured number of days when enabled")
        void should_purge_when_enabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            when(faqService.purgeInactiveOlderThan(180)).thenReturn(7);

            cronJob.purgeInactiveFaqs();

            verify(faqService).purgeInactiveOlderThan(180);
        }

        @Test
        @DisplayName("should skip when disabled")
        void should_skip_when_disabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", false);

            cronJob.purgeInactiveFaqs();

            verifyNoInteractions(faqService);
        }

        @Test
        @DisplayName("should catch and log exceptions without rethrowing")
        void should_catch_and_log_exceptions_without_rethrowing() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            doThrow(new RuntimeException("DB connection failed"))
                    .when(faqService).purgeInactiveOlderThan(180);

            cronJob.purgeInactiveFaqs();

            verify(faqService).purgeInactiveOlderThan(180);
        }
    }
}
