package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.repositories.SupportTicketRepository;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link SupportTicketService#closeStaleResolvedTickets(int)}.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketService closeStaleResolvedTickets Unit Tests")
class SupportTicketServiceStaleCloseUnitTest {

    @Mock
    private SupportTicketRepository ticketRepository;

    private SupportTicketService service;

    @BeforeEach
    void setUp() {
        service = new SupportTicketService(
                ticketRepository, null, null, null, null, null, null, null, null, null);
    }

    @Test
    @DisplayName("should close every stale resolved ticket and return the count")
    void should_close_stale_resolved_tickets_and_return_count() {
        SupportTicket ticket1 = buildResolvedTicket(1L, LocalDateTime.now().minusDays(20));
        SupportTicket ticket2 = buildResolvedTicket(2L, LocalDateTime.now().minusDays(45));

        when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any(LocalDateTime.class)))
                .thenReturn(List.of(ticket1, ticket2));

        int closed = service.closeStaleResolvedTickets(14);

        assertThat(closed).isEqualTo(2);
        assertThat(ticket1.getStatus()).isEqualTo(TicketStatus.CLOSED);
        assertThat(ticket2.getStatus()).isEqualTo(TicketStatus.CLOSED);
        verify(ticketRepository).save(ticket1);
        verify(ticketRepository).save(ticket2);
    }

    @Test
    @DisplayName("should compute the cutoff as now minus the given number of days")
    void should_compute_cutoff_from_olderThanDays() {
        when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any(LocalDateTime.class)))
                .thenReturn(List.of());

        LocalDateTime before = LocalDateTime.now().minusDays(30);

        service.closeStaleResolvedTickets(30);

        LocalDateTime after = LocalDateTime.now().minusDays(30);

        ArgumentCaptor<LocalDateTime> cutoffCaptor = ArgumentCaptor.forClass(LocalDateTime.class);
        verify(ticketRepository).findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), cutoffCaptor.capture());

        LocalDateTime cutoff = cutoffCaptor.getValue();
        assertThat(cutoff).isBetween(before.minusSeconds(2), after.plusSeconds(2));
    }

    @Test
    @DisplayName("should return zero and save nothing when there are no stale tickets")
    void should_return_zero_when_no_stale_tickets() {
        when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any(LocalDateTime.class)))
                .thenReturn(List.of());

        int closed = service.closeStaleResolvedTickets(14);

        assertThat(closed).isEqualTo(0);
        verify(ticketRepository, never()).save(any());
    }

    private SupportTicket buildResolvedTicket(Long id, LocalDateTime resolvedTime) {
        SupportTicket ticket = new SupportTicket();
        ticket.setId(id);
        ticket.setStatus(TicketStatus.RESOLVED);
        ticket.setResolvedTime(resolvedTime);
        return ticket;
    }
}
