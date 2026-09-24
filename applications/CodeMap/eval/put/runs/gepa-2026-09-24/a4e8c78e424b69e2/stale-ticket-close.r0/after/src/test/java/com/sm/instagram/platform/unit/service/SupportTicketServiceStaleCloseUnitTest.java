package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.repositories.*;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import com.sm.instagram.platform.support.ticket.services.TicketAccessTokenService;
import com.sm.instagram.platform.support.ticket.services.TicketReferenceService;
import com.sm.instagram.platform.storage.service.SignedUrlService;
import com.sm.instagram.platform.user.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

/**
 * Unit tests for SupportTicketService#closeStaleResolvedTickets.
 */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
@DisplayName("SupportTicketService closeStaleResolvedTickets Unit Tests")
class SupportTicketServiceStaleCloseUnitTest {

    @Mock
    private SupportTicketRepository ticketRepository;

    @Mock
    private TicketResponseRepository responseRepository;

    @Mock
    private TicketAttachmentRepository ticketAttachmentRepository;

    @Mock
    private ResponseAttachmentRepository responseAttachmentRepository;

    @Mock
    private UserRepository userRepository;

    @Mock
    private TicketReferenceService referenceService;

    @Mock
    private EmailService emailService;

    @Mock
    private PermissionUtils permissionUtils;

    @Mock
    private SignedUrlService signedUrlService;

    @Mock
    private TicketAccessTokenService accessTokenService;

    private SupportTicketService service;

    @BeforeEach
    void setUp() {
        service = new SupportTicketService(
                ticketRepository,
                responseRepository,
                ticketAttachmentRepository,
                responseAttachmentRepository,
                userRepository,
                referenceService,
                emailService,
                permissionUtils,
                signedUrlService,
                accessTokenService);
    }

    private SupportTicket resolvedTicket(long id, LocalDateTime resolvedTime) {
        SupportTicket ticket = new SupportTicket();
        ticket.setId(id);
        ticket.setStatus(TicketStatus.RESOLVED);
        ticket.setResolvedTime(resolvedTime);
        return ticket;
    }

    @Test
    @DisplayName("should close every ticket resolved before the cutoff and return the count")
    void should_close_stale_resolved_tickets_and_return_count() {
        SupportTicket first = resolvedTicket(1L, LocalDateTime.now().minusDays(20));
        SupportTicket second = resolvedTicket(2L, LocalDateTime.now().minusDays(15));
        when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any()))
                .thenReturn(List.of(first, second));
        when(ticketRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

        int closed = service.closeStaleResolvedTickets(14);

        assertThat(closed).isEqualTo(2);
        assertThat(first.getStatus()).isEqualTo(TicketStatus.CLOSED);
        assertThat(second.getStatus()).isEqualTo(TicketStatus.CLOSED);
        verify(ticketRepository, times(2)).save(any());
    }

    @Test
    @DisplayName("should use a cutoff of now minus the given number of days")
    void should_use_cutoff_of_now_minus_days() {
        when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any()))
                .thenReturn(List.of());

        service.closeStaleResolvedTickets(14);

        ArgumentCaptor<LocalDateTime> cutoffCaptor = ArgumentCaptor.forClass(LocalDateTime.class);
        verify(ticketRepository).findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), cutoffCaptor.capture());

        LocalDateTime expectedCutoff = LocalDateTime.now().minusDays(14);
        assertThat(cutoffCaptor.getValue()).isCloseTo(expectedCutoff, within(2, java.time.temporal.ChronoUnit.SECONDS));
    }

    @Test
    @DisplayName("should return zero and save nothing when no ticket is stale")
    void should_return_zero_when_no_stale_ticket() {
        when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any()))
                .thenReturn(List.of());

        int closed = service.closeStaleResolvedTickets(14);

        assertThat(closed).isZero();
        verify(ticketRepository, never()).save(any());
    }
}
