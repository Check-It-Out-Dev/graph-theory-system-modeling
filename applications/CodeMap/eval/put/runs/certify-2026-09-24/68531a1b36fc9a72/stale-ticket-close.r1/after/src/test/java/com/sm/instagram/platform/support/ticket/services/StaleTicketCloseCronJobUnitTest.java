package com.sm.instagram.platform.support.ticket.services;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import static org.assertj.core.api.Assertions.assertThatCode;
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
        ReflectionTestUtils.setField(cronJob, "olderThanDays", 14);
    }

    @Nested
    @DisplayName("closeStaleTickets")
    class CloseStaleTickets {

        @Test
        @DisplayName("delegates to the service with the configured olderThanDays when enabled")
        void delegatesToServiceWhenEnabled() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            when(supportTicketService.closeStaleResolvedTickets(14)).thenReturn(3);

            cronJob.closeStaleTickets();

            verify(supportTicketService).closeStaleResolvedTickets(14);
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
        void catchesAndLogsServiceFailure() {
            ReflectionTestUtils.setField(cronJob, "enabled", true);
            doThrow(new RuntimeException("DB connection failed"))
                    .when(supportTicketService).closeStaleResolvedTickets(14);

            assertThatCode(() -> cronJob.closeStaleTickets()).doesNotThrowAnyException();

            verify(supportTicketService).closeStaleResolvedTickets(14);
        }
    }
}
