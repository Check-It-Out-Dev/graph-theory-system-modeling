package com.sm.instagram.platform.support.ticket.services;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import static org.mockito.Mockito.*;

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
        @DisplayName("should close stale tickets with the configured day threshold when enabled")
        void shouldCloseStaleTicketsWithConfiguredDaysWhenEnabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 21);
            when(supportTicketService.closeStaleResolvedTickets(21)).thenReturn(7);

            cronJob.closeStaleTickets();

            verify(supportTicketService).closeStaleResolvedTickets(21);
        }

        @Test
        @DisplayName("should do nothing when disabled")
        void shouldDoNothingWhenDisabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", false);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 14);

            cronJob.closeStaleTickets();

            verifyNoInteractions(supportTicketService);
        }

        @Test
        @DisplayName("should catch and log the service's failure without rethrowing")
        void shouldCatchAndLogServiceFailureWithoutRethrowing() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 14);
            doThrow(new RuntimeException("DB connection failed"))
                    .when(supportTicketService).closeStaleResolvedTickets(14);

            org.junit.jupiter.api.Assertions.assertDoesNotThrow(() -> cronJob.closeStaleTickets());

            verify(supportTicketService).closeStaleResolvedTickets(14);
        }
    }
}
