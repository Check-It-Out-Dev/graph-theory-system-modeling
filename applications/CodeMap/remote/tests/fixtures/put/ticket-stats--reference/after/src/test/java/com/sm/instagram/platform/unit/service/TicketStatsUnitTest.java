package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.common.exceptions.InsufficientPermissionsException;
import com.sm.instagram.platform.common.exceptions.ValidationTranslatableException;
import com.sm.instagram.platform.support.ticket.dtos.TicketStatsDtoOut;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.repositories.SupportTicketRepository;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDate;
import java.time.LocalDateTime;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class TicketStatsUnitTest {

    @Mock
    private SupportTicketRepository ticketRepository;

    @Mock
    private PermissionUtils permissionUtils;

    @InjectMocks
    private SupportTicketService service;

    @Test
    void countsEveryStatusInThePeriod() {
        when(permissionUtils.isAdmin()).thenReturn(true);
        LocalDate from = LocalDate.of(2026, 9, 1);
        LocalDate to = LocalDate.of(2026, 9, 30);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(any(), eq(from.atStartOfDay()),
                eq(LocalDateTime.of(2026, 10, 1, 0, 0)))).thenReturn(0L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(eq(TicketStatus.OPEN), eq(from.atStartOfDay()),
                eq(LocalDateTime.of(2026, 10, 1, 0, 0)))).thenReturn(4L);

        TicketStatsDtoOut stats = service.getTicketStats(from, to);

        assertThat(stats.getCountsByStatus()).hasSize(TicketStatus.values().length).containsEntry(TicketStatus.OPEN, 4L);
        assertThat(stats.getTotal()).isEqualTo(4L);
    }

    @Test
    void rejectsAReversedPeriod() {
        when(permissionUtils.isAdmin()).thenReturn(true);
        assertThatThrownBy(() -> service.getTicketStats(LocalDate.of(2026, 9, 2), LocalDate.of(2026, 9, 1)))
                .isInstanceOf(ValidationTranslatableException.class)
                .hasMessage("error.validation.invalid_date_range");
    }

    @Test
    void refusesANonAdmin() {
        when(permissionUtils.isAdmin()).thenReturn(false);
        assertThatThrownBy(() -> service.getTicketStats(LocalDate.of(2026, 9, 1), LocalDate.of(2026, 9, 2)))
                .isInstanceOf(InsufficientPermissionsException.class);
    }
}
