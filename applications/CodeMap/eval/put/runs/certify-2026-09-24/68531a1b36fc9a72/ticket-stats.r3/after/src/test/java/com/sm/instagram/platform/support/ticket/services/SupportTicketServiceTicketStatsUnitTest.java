package com.sm.instagram.platform.support.ticket.services;

import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.common.exceptions.InsufficientPermissionsException;
import com.sm.instagram.platform.common.exceptions.ValidationTranslatableException;
import com.sm.instagram.platform.support.ticket.dtos.TicketStatsDtoOut;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.repositories.SupportTicketRepository;
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
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link SupportTicketService#getTicketStats(LocalDate, LocalDate)}.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketService#getTicketStats")
class SupportTicketServiceTicketStatsUnitTest {

    @Mock
    private SupportTicketRepository ticketRepository;

    @Mock
    private PermissionUtils permissionUtils;

    private SupportTicketService service;

    @BeforeEach
    void setUp() {
        service = new SupportTicketService(
                ticketRepository, null, null, null, null, null, null, permissionUtils, null, null);
    }

    @Test
    @DisplayName("counts every status over the requested period and sums the total")
    void countsEveryStatusOverTheRequestedPeriodAndSumsTheTotal() {
        LocalDate from = LocalDate.of(2026, 1, 1);
        LocalDate to = LocalDate.of(2026, 1, 31);
        LocalDateTime expectedStart = from.atStartOfDay();
        LocalDateTime expectedEnd = to.plusDays(1).atStartOfDay();

        when(permissionUtils.isAdmin()).thenReturn(true);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.OPEN, expectedStart, expectedEnd)).thenReturn(3L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.IN_PROGRESS, expectedStart, expectedEnd)).thenReturn(2L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.WAITING_FOR_CUSTOMER, expectedStart, expectedEnd)).thenReturn(0L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.RESOLVED, expectedStart, expectedEnd)).thenReturn(5L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.CLOSED, expectedStart, expectedEnd)).thenReturn(1L);

        TicketStatsDtoOut result = service.getTicketStats(from, to);

        assertThat(result.getCountsByStatus())
                .containsEntry(TicketStatus.OPEN, 3L)
                .containsEntry(TicketStatus.IN_PROGRESS, 2L)
                .containsEntry(TicketStatus.WAITING_FOR_CUSTOMER, 0L)
                .containsEntry(TicketStatus.RESOLVED, 5L)
                .containsEntry(TicketStatus.CLOSED, 1L);
        assertThat(result.getTotal()).isEqualTo(11L);
        verify(ticketRepository).countByStatusAndCreatedTimeBetween(TicketStatus.OPEN, expectedStart, expectedEnd);
        verify(ticketRepository).countByStatusAndCreatedTimeBetween(TicketStatus.CLOSED, expectedStart, expectedEnd);
    }

    @Test
    @DisplayName("returns zero for every status when nothing was created in the period")
    void zeroesEveryStatusWhenNoTicketMatches() {
        LocalDate from = LocalDate.of(2026, 2, 1);
        LocalDate to = LocalDate.of(2026, 2, 1);

        when(permissionUtils.isAdmin()).thenReturn(true);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(any(), any(), any())).thenReturn(0L);

        TicketStatsDtoOut result = service.getTicketStats(from, to);

        assertThat(result.getCountsByStatus()).hasSize(TicketStatus.values().length);
        assertThat(result.getCountsByStatus().values()).allMatch(count -> count == 0L);
        assertThat(result.getTotal()).isZero();
    }

    @Test
    @DisplayName("rejects a range where from is after to")
    void rejectsARangeWhereFromIsAfterTo() {
        when(permissionUtils.isAdmin()).thenReturn(true);

        LocalDate from = LocalDate.of(2026, 3, 10);
        LocalDate to = LocalDate.of(2026, 3, 1);

        assertThatThrownBy(() -> service.getTicketStats(from, to))
                .isInstanceOf(ValidationTranslatableException.class)
                .hasFieldOrPropertyWithValue("messageKey", "error.validation.invalid_date_range");
        verifyNoInteractions(ticketRepository);
    }

    @Test
    @DisplayName("denies a non-admin with the same error updateTicketStatus gives")
    void deniesANonAdminWithTheSameErrorAsUpdateTicketStatus() {
        when(permissionUtils.getUserId()).thenReturn("user-uid");
        when(permissionUtils.isAdmin()).thenReturn(false);

        LocalDate from = LocalDate.of(2026, 1, 1);
        LocalDate to = LocalDate.of(2026, 1, 31);

        assertThatThrownBy(() -> service.getTicketStats(from, to))
                .isInstanceOf(InsufficientPermissionsException.class)
                .hasFieldOrPropertyWithValue("messageKey", "error.auth.insufficient_permissions");
        verifyNoInteractions(ticketRepository);
    }
}
