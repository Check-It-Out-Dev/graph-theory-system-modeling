package com.sm.instagram.platform.support.ticket.services;

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
import com.sm.instagram.platform.storage.service.SignedUrlService;
import com.sm.instagram.platform.user.UserRepository;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link SupportTicketService#getTicketStats(LocalDate, LocalDate)}.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketService#getTicketStats Unit Tests")
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

    @InjectMocks
    private SupportTicketService service;

    private static final LocalDate FROM = LocalDate.of(2026, 1, 10);
    private static final LocalDate TO = LocalDate.of(2026, 1, 15);
    private static final LocalDateTime START = FROM.atStartOfDay();
    private static final LocalDateTime END = TO.plusDays(1).atStartOfDay();

    @Test
    @DisplayName("counts every status and sums the total for an admin caller")
    void countsEveryStatusAndSumsTheTotalForAdmin() {
        when(permissionUtils.getUserId()).thenReturn("admin-uid");
        when(permissionUtils.isAdmin()).thenReturn(true);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.OPEN, START, END)).thenReturn(3L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.IN_PROGRESS, START, END)).thenReturn(1L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.WAITING_FOR_CUSTOMER, START, END)).thenReturn(0L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.RESOLVED, START, END)).thenReturn(2L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.CLOSED, START, END)).thenReturn(4L);

        TicketStatsDtoOut result = service.getTicketStats(FROM, TO);

        Map<TicketStatus, Long> counts = result.getCountsByStatus();
        assertThat(counts).hasSize(TicketStatus.values().length);
        assertThat(counts.get(TicketStatus.OPEN)).isEqualTo(3L);
        assertThat(counts.get(TicketStatus.IN_PROGRESS)).isEqualTo(1L);
        assertThat(counts.get(TicketStatus.WAITING_FOR_CUSTOMER)).isEqualTo(0L);
        assertThat(counts.get(TicketStatus.RESOLVED)).isEqualTo(2L);
        assertThat(counts.get(TicketStatus.CLOSED)).isEqualTo(4L);
        assertThat(result.getTotal()).isEqualTo(10L);

        verify(ticketRepository).countByStatusAndCreatedTimeBetween(TicketStatus.OPEN, START, END);
        verify(ticketRepository).countByStatusAndCreatedTimeBetween(TicketStatus.CLOSED, START, END);
    }

    @Test
    @DisplayName("reports zero for every status when no ticket was created in the period")
    void reportsZeroForEveryStatusWhenNoTicketsExist() {
        when(permissionUtils.getUserId()).thenReturn("admin-uid");
        when(permissionUtils.isAdmin()).thenReturn(true);
        // No stubbing of countByStatusAndCreatedTimeBetween: the mock's default
        // 0L return is exactly the "0 when none" outcome the interface requires.

        TicketStatsDtoOut result = service.getTicketStats(FROM, TO);

        assertThat(result.getCountsByStatus()).hasSize(TicketStatus.values().length);
        assertThat(result.getCountsByStatus().values()).allMatch(count -> count == 0L);
        assertThat(result.getTotal()).isEqualTo(0L);

        for (TicketStatus status : TicketStatus.values()) {
            verify(ticketRepository).countByStatusAndCreatedTimeBetween(status, START, END);
        }
    }

    @Test
    @DisplayName("treats a single-day range as from's start of day to the day after")
    void treatsASingleDayRangeAsFromsStartOfDayToTheDayAfter() {
        LocalDate day = LocalDate.of(2026, 3, 1);
        LocalDateTime dayStart = day.atStartOfDay();
        LocalDateTime dayEnd = day.plusDays(1).atStartOfDay();

        when(permissionUtils.getUserId()).thenReturn("admin-uid");
        when(permissionUtils.isAdmin()).thenReturn(true);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.OPEN, dayStart, dayEnd)).thenReturn(5L);

        TicketStatsDtoOut result = service.getTicketStats(day, day);

        assertThat(result.getCountsByStatus().get(TicketStatus.OPEN)).isEqualTo(5L);
        assertThat(result.getTotal()).isEqualTo(5L);
        verify(ticketRepository).countByStatusAndCreatedTimeBetween(TicketStatus.OPEN, dayStart, dayEnd);
    }

    @Test
    @DisplayName("rejects a from date after the to date without querying the repository")
    void rejectsAFromDateAfterTheToDate() {
        when(permissionUtils.getUserId()).thenReturn("admin-uid");
        when(permissionUtils.isAdmin()).thenReturn(true);

        assertThatThrownBy(() -> service.getTicketStats(TO, FROM))
                .isInstanceOf(ValidationTranslatableException.class)
                .hasFieldOrPropertyWithValue("messageKey", "error.validation.invalid_date_range");

        verifyNoInteractions(ticketRepository);
    }

    @Test
    @DisplayName("denies a non-admin caller with the same error updateTicketStatus gives")
    void deniesANonAdminCaller() {
        when(permissionUtils.getUserId()).thenReturn("user-uid");
        when(permissionUtils.isAdmin()).thenReturn(false);

        assertThatThrownBy(() -> service.getTicketStats(FROM, TO))
                .isInstanceOf(InsufficientPermissionsException.class)
                .hasFieldOrPropertyWithValue("messageKey", "error.auth.insufficient_permissions");

        verifyNoInteractions(ticketRepository);
    }
}
