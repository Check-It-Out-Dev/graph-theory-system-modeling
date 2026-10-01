package com.sm.instagram.platform.support.ticket.services;

import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.storage.service.SignedUrlService;
import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.repositories.ResponseAttachmentRepository;
import com.sm.instagram.platform.support.ticket.repositories.SupportTicketRepository;
import com.sm.instagram.platform.support.ticket.repositories.TicketAttachmentRepository;
import com.sm.instagram.platform.support.ticket.repositories.TicketResponseRepository;
import com.sm.instagram.platform.user.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketService.closeStaleResolvedTickets Unit Tests")
class SupportTicketServiceCloseStaleResolvedTicketsUnitTest {

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

    @Test
    @DisplayName("should close every stale resolved ticket and return the count")
    void shouldCloseEveryStaleResolvedTicketAndReturnCount() {
        SupportTicket ticket1 = new SupportTicket();
        ticket1.setId(11L);
        ticket1.setStatus(TicketStatus.RESOLVED);

        SupportTicket ticket2 = new SupportTicket();
        ticket2.setId(22L);
        ticket2.setStatus(TicketStatus.RESOLVED);

        when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any()))
                .thenReturn(List.of(ticket1, ticket2));
        when(ticketRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

        int closed = service.closeStaleResolvedTickets(21);

        assertThat(closed).isEqualTo(2);
        assertThat(ticket1.getStatus()).isEqualTo(TicketStatus.CLOSED);
        assertThat(ticket2.getStatus()).isEqualTo(TicketStatus.CLOSED);
        verify(ticketRepository).save(ticket1);
        verify(ticketRepository).save(ticket2);
    }

    @Test
    @DisplayName("should query with a cutoff of now minus the given number of days")
    void shouldQueryWithCutoffOfNowMinusGivenDays() {
        when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any()))
                .thenReturn(Collections.emptyList());

        LocalDateTime before = LocalDateTime.now().minusDays(21);
        service.closeStaleResolvedTickets(21);
        LocalDateTime after = LocalDateTime.now().minusDays(21);

        ArgumentCaptor<LocalDateTime> cutoffCaptor = ArgumentCaptor.forClass(LocalDateTime.class);
        verify(ticketRepository).findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), cutoffCaptor.capture());

        LocalDateTime cutoff = cutoffCaptor.getValue();
        assertThat(cutoff).isBetween(before.minus(2, ChronoUnit.SECONDS), after.plus(2, ChronoUnit.SECONDS));
    }

    @Test
    @DisplayName("should return zero and save nothing when there are no stale resolved tickets")
    void shouldReturnZeroWhenNoStaleResolvedTickets() {
        when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any()))
                .thenReturn(Collections.emptyList());

        int closed = service.closeStaleResolvedTickets(14);

        assertThat(closed).isZero();
        verify(ticketRepository, never()).save(any());
    }
}
