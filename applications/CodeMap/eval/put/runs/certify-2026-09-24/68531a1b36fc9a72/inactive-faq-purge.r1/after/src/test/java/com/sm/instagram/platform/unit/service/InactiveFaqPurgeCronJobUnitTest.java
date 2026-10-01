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

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

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
        @DisplayName("should purge with the configured retention when enabled")
        void should_purge_with_configured_retention_when_enabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 45);
            when(faqService.purgeInactiveOlderThan(45)).thenReturn(7);

            cronJob.purgeInactiveFaqs();

            verify(faqService).purgeInactiveOlderThan(45);
        }

        @Test
        @DisplayName("should skip when disabled")
        void should_skip_when_disabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", false);

            cronJob.purgeInactiveFaqs();

            verifyNoInteractions(faqService);
        }

        @Test
        @DisplayName("should catch and log a failure without rethrowing")
        void should_catch_and_log_exceptions_without_rethrowing() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 180);
            doThrow(new RuntimeException("DB connection failed"))
                    .when(faqService).purgeInactiveOlderThan(180);

            assertDoesNotThrow(() -> cronJob.purgeInactiveFaqs());

            verify(faqService).purgeInactiveOlderThan(180);
        }
    }
}
