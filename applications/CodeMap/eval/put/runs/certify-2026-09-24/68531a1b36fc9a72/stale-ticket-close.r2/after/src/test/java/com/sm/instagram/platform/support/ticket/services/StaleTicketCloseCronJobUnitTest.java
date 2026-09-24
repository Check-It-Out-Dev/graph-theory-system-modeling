package com.sm.instagram.platform.support.ticket.services;

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

/**
 * Unit test for {@link StaleTicketCloseCronJob}.
 */
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
        @DisplayName("delegates to the service with the configured day count when enabled")
        void delegatesToServiceWhenEnabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 21);
            when(supportTicketService.closeStaleResolvedTickets(21)).thenReturn(3);

            cronJob.closeStaleTickets();

            verify(supportTicketService).closeStaleResolvedTickets(21);
        }

        @Test
        @DisplayName("does nothing when disabled")
        void doesNothingWhenDisabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", false);

            cronJob.closeStaleTickets();

            verifyNoInteractions(supportTicketService);
        }

        @Test
        @DisplayName("catches and logs a service failure instead of rethrowing")
        void catchesServiceFailure() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            ReflectionTestUtils.setField(cronJob, "olderThanDays", 14);
            doThrow(new RuntimeException("db down")).when(supportTicketService).closeStaleResolvedTickets(14);

            assertDoesNotThrow(() -> cronJob.closeStaleTickets());

            verify(supportTicketService).closeStaleResolvedTickets(14);
        }
    }
}
