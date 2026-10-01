package com.sm.instagram.platform.support.ticket.services;

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
@DisplayName("StaleTicketCloseCronJob Unit Tests")
class StaleTicketCloseCronJobUnitTest {

    @Mock
    private SupportTicketService supportTicketService;

    private StaleTicketCloseCronJob cronJob;

    @BeforeEach
    void setUp() {
        cronJob = new StaleTicketCloseCronJob(supportTicketService);
        ReflectionTestUtils.setField(cronJob, "olderThanDays", 14);
    }

    @Nested
    @DisplayName("closeStaleTickets")
    class CloseStaleTickets {

        @Test
        @DisplayName("should close stale tickets with the configured threshold when enabled")
        void should_closeStaleTickets_when_enabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            when(supportTicketService.closeStaleResolvedTickets(14)).thenReturn(3);

            cronJob.closeStaleTickets();

            verify(supportTicketService).closeStaleResolvedTickets(14);
        }

        @Test
        @DisplayName("should skip when disabled")
        void should_skip_when_disabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", false);

            cronJob.closeStaleTickets();

            verifyNoInteractions(supportTicketService);
        }

        @Test
        @DisplayName("should catch and log exceptions without rethrowing")
        void should_catch_and_log_exceptions_without_rethrowing() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            doThrow(new RuntimeException("DB connection failed"))
                    .when(supportTicketService).closeStaleResolvedTickets(14);

            // Should not throw
            cronJob.closeStaleTickets();

            verify(supportTicketService).closeStaleResolvedTickets(14);
        }
    }
}
