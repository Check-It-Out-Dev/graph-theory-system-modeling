package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.common.exceptions.InsufficientPermissionsException;
import com.sm.instagram.platform.common.exceptions.ValidationTranslatableException;
import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.dtos.TicketStatsDtoOut;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.repositories.ResponseAttachmentRepository;
import com.sm.instagram.platform.support.ticket.repositories.SupportTicketRepository;
import com.sm.instagram.platform.support.ticket.repositories.TicketAttachmentRepository;
import com.sm.instagram.platform.support.ticket.repositories.TicketResponseRepository;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import com.sm.instagram.platform.support.ticket.services.TicketAccessTokenService;
import com.sm.instagram.platform.support.ticket.services.TicketReferenceService;
import com.sm.instagram.platform.storage.service.SignedUrlService;
import com.sm.instagram.platform.user.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDate;
import java.time.LocalDateTime;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link SupportTicketService#getTicketStats(LocalDate, LocalDate)}.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketService.getTicketStats Unit Tests")
class SupportTicketServiceStatsUnitTest {

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
    @DisplayName("returns per-status counts and their total for the period")
    void returnsCountsByStatusAndTotal() {
        when(permissionUtils.getUserId()).thenReturn("admin-firebase-uid");
        when(permissionUtils.isAdmin()).thenReturn(true);

        LocalDate from = LocalDate.of(2026, 1, 5);
        LocalDate to = LocalDate.of(2026, 1, 10);
        LocalDateTime expectedStart = from.atStartOfDay();
        LocalDateTime expectedEnd = to.plusDays(1).atStartOfDay();

        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.OPEN, expectedStart, expectedEnd)).thenReturn(3L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.IN_PROGRESS, expectedStart, expectedEnd)).thenReturn(2L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.WAITING_FOR_CUSTOMER, expectedStart, expectedEnd)).thenReturn(0L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.RESOLVED, expectedStart, expectedEnd)).thenReturn(5L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.CLOSED, expectedStart, expectedEnd)).thenReturn(1L);

        TicketStatsDtoOut stats = service.getTicketStats(from, to);

        assertThat(stats.getCountsByStatus()).hasSize(TicketStatus.values().length);
        assertThat(stats.getCountsByStatus().get(TicketStatus.OPEN)).isEqualTo(3L);
        assertThat(stats.getCountsByStatus().get(TicketStatus.IN_PROGRESS)).isEqualTo(2L);
        assertThat(stats.getCountsByStatus().get(TicketStatus.WAITING_FOR_CUSTOMER)).isEqualTo(0L);
        assertThat(stats.getCountsByStatus().get(TicketStatus.RESOLVED)).isEqualTo(5L);
        assertThat(stats.getCountsByStatus().get(TicketStatus.CLOSED)).isEqualTo(1L);
        assertThat(stats.getTotal()).isEqualTo(11L);

        for (TicketStatus status : TicketStatus.values()) {
            verify(ticketRepository).countByStatusAndCreatedTimeBetween(eq(status), eq(expectedStart), eq(expectedEnd));
        }
    }

    @Test
    @DisplayName("counts every status as 0 when nothing was created in the period")
    void returnsZeroForEveryStatusWhenNoneCreated() {
        when(permissionUtils.getUserId()).thenReturn("admin-firebase-uid");
        when(permissionUtils.isAdmin()).thenReturn(true);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(any(), any(), any()))
                .thenReturn(0L);

        LocalDate from = LocalDate.of(2026, 2, 1);
        LocalDate to = LocalDate.of(2026, 2, 1);

        TicketStatsDtoOut stats = service.getTicketStats(from, to);

        assertThat(stats.getCountsByStatus()).hasSize(TicketStatus.values().length);
        assertThat(stats.getCountsByStatus().values()).allMatch(count -> count == 0L);
        assertThat(stats.getTotal()).isZero();
    }

    @Test
    @DisplayName("rejects a period where from is after to")
    void rejectsInvertedDateRange() {
        when(permissionUtils.getUserId()).thenReturn("admin-firebase-uid");
        when(permissionUtils.isAdmin()).thenReturn(true);

        LocalDate from = LocalDate.of(2026, 3, 10);
        LocalDate to = LocalDate.of(2026, 3, 1);

        assertThatThrownBy(() -> service.getTicketStats(from, to))
                .isInstanceOf(ValidationTranslatableException.class)
                .extracting(ex -> ((ValidationTranslatableException) ex).getMessageKey())
                .isEqualTo("error.validation.invalid_date_range");

        verify(ticketRepository, never())
                .countByStatusAndCreatedTimeBetween(any(), any(), any());
    }

    @Test
    @DisplayName("rejects a non-administrator with the same error updateTicketStatus gives")
    void rejectsNonAdministrator() {
        when(permissionUtils.getUserId()).thenReturn("regular-firebase-uid");
        when(permissionUtils.isAdmin()).thenReturn(false);

        LocalDate from = LocalDate.of(2026, 1, 1);
        LocalDate to = LocalDate.of(2026, 1, 31);

        assertThatThrownBy(() -> service.getTicketStats(from, to))
                .isInstanceOf(InsufficientPermissionsException.class)
                .extracting(ex -> ((InsufficientPermissionsException) ex).getMessageKey())
                .isEqualTo("error.auth.insufficient_permissions");

        verify(ticketRepository, never())
                .countByStatusAndCreatedTimeBetween(any(), any(), any());
    }
}
