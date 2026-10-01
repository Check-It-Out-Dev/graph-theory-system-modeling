package com.sm.instagram.platform.support.ticket.services;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doThrow;
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
        @DisplayName("should close stale tickets using the configured day threshold when enabled")
        void shouldCloseStaleTicketsWhenEnabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 21);
            when(supportTicketService.closeStaleResolvedTickets(21)).thenReturn(3);

            cronJob.closeStaleTickets();

            verify(supportTicketService).closeStaleResolvedTickets(21);
        }

        @Test
        @DisplayName("should skip when disabled")
        void shouldSkipWhenDisabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", false);

            cronJob.closeStaleTickets();

            verifyNoInteractions(supportTicketService);
        }

        @Test
        @DisplayName("should catch and log exceptions without rethrowing")
        void shouldCatchAndLogExceptionsWithoutRethrowing() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 14);
            doThrow(new RuntimeException("DB connection failed"))
                    .when(supportTicketService).closeStaleResolvedTickets(anyInt());

            cronJob.closeStaleTickets();

            verify(supportTicketService).closeStaleResolvedTickets(14);
        }
    }
}
