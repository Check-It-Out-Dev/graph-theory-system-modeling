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
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit test for {@link SupportTicketService#closeStaleResolvedTickets(int)} —
 * the nightly cleanup that closes RESOLVED tickets nobody reopened.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketService.closeStaleResolvedTickets")
class SupportTicketServiceStaleTicketCloseUnitTest {

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
    @DisplayName("closes every RESOLVED ticket past the cutoff and returns the count")
    void closesStaleResolvedTickets() {
        SupportTicket first = new SupportTicket();
        first.setId(11L);
        first.setStatus(TicketStatus.RESOLVED);
        SupportTicket second = new SupportTicket();
        second.setId(12L);
        second.setStatus(TicketStatus.RESOLVED);

        when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any(LocalDateTime.class)))
                .thenReturn(List.of(first, second));
        when(ticketRepository.save(any(SupportTicket.class))).thenAnswer(inv -> inv.getArgument(0));

        int closed = service.closeStaleResolvedTickets(21);

        assertEquals(2, closed);
        assertEquals(TicketStatus.CLOSED, first.getStatus());
        assertEquals(TicketStatus.CLOSED, second.getStatus());
        verify(ticketRepository).save(first);
        verify(ticketRepository).save(second);
    }

    @Test
    @DisplayName("uses now-minus-days as the cutoff passed to the repository")
    void usesConfiguredDaysAsCutoff() {
        when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any(LocalDateTime.class)))
                .thenReturn(List.of());

        LocalDateTime before = LocalDateTime.now().minusDays(30);
        service.closeStaleResolvedTickets(30);
        LocalDateTime after = LocalDateTime.now().minusDays(30);

        ArgumentCaptor<LocalDateTime> cutoffCaptor = ArgumentCaptor.forClass(LocalDateTime.class);
        verify(ticketRepository).findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), cutoffCaptor.capture());
        LocalDateTime cutoff = cutoffCaptor.getValue();

        assertFalse(cutoff.isBefore(before.minusSeconds(2)));
        assertFalse(cutoff.isAfter(after.plusSeconds(2)));
    }

    @Test
    @DisplayName("returns zero and saves nothing when no ticket is stale")
    void returnsZeroWhenNoneStale() {
        when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any(LocalDateTime.class)))
                .thenReturn(List.of());

        int closed = service.closeStaleResolvedTickets(14);

        assertEquals(0, closed);
        verify(ticketRepository, never()).save(any());
    }
}
