package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.ticket.services.StaleTicketCloseCronJob;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@DisplayName("StaleTicketCloseCronJob Unit Tests")
class StaleTicketCloseCronJobUnitTest {

    @Mock
    private SupportTicketService supportTicketService;

    private StaleTicketCloseCronJob cronJob;

    @BeforeEach
    void setUp() {
        cronJob = new StaleTicketCloseCronJob(supportTicketService);
    }

    @Nested
    @DisplayName("closeStaleTickets")
    class CloseStaleTickets {

        @Test
        @DisplayName("should delegate to the service with the configured olderThanDays when enabled")
        void should_delegate_to_service_when_enabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 21);
            when(supportTicketService.closeStaleResolvedTickets(21)).thenReturn(3);

            cronJob.closeStaleTickets();

            verify(supportTicketService).closeStaleResolvedTickets(21);
        }

        @Test
        @DisplayName("should do nothing when disabled")
        void should_skip_when_disabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", false);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 14);

            cronJob.closeStaleTickets();

            verifyNoInteractions(supportTicketService);
        }

        @Test
        @DisplayName("should catch and log a service failure without rethrowing")
        void should_catch_and_log_exceptions_without_rethrowing() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 14);
            when(supportTicketService.closeStaleResolvedTickets(14))
                    .thenThrow(new RuntimeException("DB connection failed"));

            assertThatCode(() -> cronJob.closeStaleTickets()).doesNotThrowAnyException();

            verify(supportTicketService).closeStaleResolvedTickets(14);
        }
    }
}
