package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.jobs.StaleTicketCloseCronJob;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class StaleTicketCloseCronJobUnitTest {

    @Mock
    private SupportTicketService supportTicketService;

    @InjectMocks
    private StaleTicketCloseCronJob job;

    @BeforeEach
    void setUp() {
        ReflectionTestUtils.setField(job, "enabled", true);
        ReflectionTestUtils.setField(job, "olderThanDays", 14);
    }

    @Test
    void closesTicketsResolvedLongerThanTheConfiguredDays() {
        when(supportTicketService.closeStaleResolvedTickets(14)).thenReturn(2);
        job.closeStaleTickets();
        verify(supportTicketService).closeStaleResolvedTickets(14);
    }

    @Test
    void doesNothingWhenDisabled() {
        ReflectionTestUtils.setField(job, "enabled", false);
        job.closeStaleTickets();
        verify(supportTicketService, never()).closeStaleResolvedTickets(anyInt());
    }

    @Test
    void swallowsServiceFailures() {
        when(supportTicketService.closeStaleResolvedTickets(14)).thenThrow(new IllegalStateException("boom"));
        assertThatCode(() -> job.closeStaleTickets()).doesNotThrowAnyException();
    }
}
