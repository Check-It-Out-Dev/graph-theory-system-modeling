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
import org.springframework.test.util.ReflectionTestUtils;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("InactiveFaqPurgeCronJob Unit Tests")
class InactiveFaqPurgeCronJobUnitTest {

    @Mock
    private FaqService faqService;

    private InactiveFaqPurgeCronJob cronJob;

    @BeforeEach
    void setUp() {
        cronJob = new InactiveFaqPurgeCronJob(faqService);
    }

    @Nested
    @DisplayName("purgeInactiveFaqs")
    class PurgeInactiveFaqs {

        @Test
        @DisplayName("should purge with the configured number of days when enabled")
        void should_purge_with_configured_days_when_enabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 200);
            when(faqService.purgeInactiveOlderThan(200)).thenReturn(7);

            cronJob.purgeInactiveFaqs();

            verify(faqService).purgeInactiveOlderThan(200);
        }

        @Test
        @DisplayName("should do nothing when disabled")
        void should_do_nothing_when_disabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", false);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 180);

            cronJob.purgeInactiveFaqs();

            verifyNoInteractions(faqService);
        }

        @Test
        @DisplayName("should catch and log a failure of the service without rethrowing")
        void should_catch_and_log_failure_without_rethrowing() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 180);
            doThrow(new RuntimeException("DB connection failed"))
                    .when(faqService).purgeInactiveOlderThan(180);

            assertThatCode(() -> cronJob.purgeInactiveFaqs()).doesNotThrowAnyException();

            verify(faqService).purgeInactiveOlderThan(180);
        }
    }
}
