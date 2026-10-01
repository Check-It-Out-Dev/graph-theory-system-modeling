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
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
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
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link SupportTicketService#closeStaleResolvedTickets(int)}.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketService.closeStaleResolvedTickets Unit Tests")
class SupportTicketServiceUnitTest {

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

    @InjectMocks
    private SupportTicketService service;

    private SupportTicket staleTicketOne;
    private SupportTicket staleTicketTwo;

    @BeforeEach
    void setUp() {
        staleTicketOne = new SupportTicket();
        staleTicketOne.setId(11L);
        staleTicketOne.setStatus(TicketStatus.RESOLVED);
        staleTicketOne.setResolvedTime(LocalDateTime.now().minusDays(20));

        staleTicketTwo = new SupportTicket();
        staleTicketTwo.setId(22L);
        staleTicketTwo.setStatus(TicketStatus.RESOLVED);
        staleTicketTwo.setResolvedTime(LocalDateTime.now().minusDays(30));
    }

    @Nested
    @DisplayName("closeStaleResolvedTickets")
    class CloseStaleResolvedTickets {

        @Test
        @DisplayName("closes and saves every stale resolved ticket, returns the count")
        void closesAndSavesEveryStaleResolvedTicket() {
            when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any(LocalDateTime.class)))
                    .thenReturn(List.of(staleTicketOne, staleTicketTwo));

            int result = service.closeStaleResolvedTickets(14);

            assertThat(result).isEqualTo(2);
            assertThat(staleTicketOne.getStatus()).isEqualTo(TicketStatus.CLOSED);
            assertThat(staleTicketTwo.getStatus()).isEqualTo(TicketStatus.CLOSED);
            verify(ticketRepository, times(1)).save(staleTicketOne);
            verify(ticketRepository, times(1)).save(staleTicketTwo);
        }

        @Test
        @DisplayName("passes a cutoff of now minus olderThanDays to the repository")
        void passesTheCorrectCutoffToTheRepository() {
            when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any(LocalDateTime.class)))
                    .thenReturn(Collections.emptyList());

            LocalDateTime before = LocalDateTime.now().minusDays(9);
            service.closeStaleResolvedTickets(9);
            LocalDateTime after = LocalDateTime.now().minusDays(9);

            ArgumentCaptor<LocalDateTime> cutoffCaptor = ArgumentCaptor.forClass(LocalDateTime.class);
            verify(ticketRepository).findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), cutoffCaptor.capture());

            LocalDateTime capturedCutoff = cutoffCaptor.getValue();
            assertThat(capturedCutoff).isBetween(before.minus(2, ChronoUnit.SECONDS), after.plus(2, ChronoUnit.SECONDS));
        }

        @Test
        @DisplayName("returns 0 and saves nothing when no ticket is stale")
        void returnsZeroWhenNoTicketIsStale() {
            when(ticketRepository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any(LocalDateTime.class)))
                    .thenReturn(Collections.emptyList());

            int result = service.closeStaleResolvedTickets(14);

            assertThat(result).isEqualTo(0);
            verify(ticketRepository, never()).save(any());
        }
    }
}
