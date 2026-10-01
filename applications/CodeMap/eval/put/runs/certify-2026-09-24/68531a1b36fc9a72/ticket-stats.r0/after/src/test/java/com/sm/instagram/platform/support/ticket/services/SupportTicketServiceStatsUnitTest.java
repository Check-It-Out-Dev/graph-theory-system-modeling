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
@DisplayName("SupportTicketService#getTicketStats")
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
    @DisplayName("denies a non-admin caller with the same error updateTicketStatus gives")
    void deniesNonAdminCaller() {
        when(permissionUtils.getUserId()).thenReturn("user-uid");
        when(permissionUtils.isAdmin()).thenReturn(false);

        assertThatThrownBy(() -> service.getTicketStats(LocalDate.of(2026, 1, 1), LocalDate.of(2026, 1, 31)))
                .isInstanceOf(InsufficientPermissionsException.class)
                .hasFieldOrPropertyWithValue("messageKey", "error.auth.insufficient_permissions");

        verify(ticketRepository, never())
                .countByStatusAndCreatedTimeBetween(any(TicketStatus.class), any(LocalDateTime.class), any(LocalDateTime.class));
    }

    @Test
    @DisplayName("rejects a range where from is after to")
    void rejectsInvertedRange() {
        when(permissionUtils.getUserId()).thenReturn("admin-uid");
        when(permissionUtils.isAdmin()).thenReturn(true);

        assertThatThrownBy(() -> service.getTicketStats(LocalDate.of(2026, 2, 10), LocalDate.of(2026, 2, 1)))
                .isInstanceOf(ValidationTranslatableException.class)
                .hasFieldOrPropertyWithValue("messageKey", "error.validation.invalid_date_range");

        verify(ticketRepository, never())
                .countByStatusAndCreatedTimeBetween(any(TicketStatus.class), any(LocalDateTime.class), any(LocalDateTime.class));
    }

    @Test
    @DisplayName("counts every status with the atStartOfDay/plusDays(1) window and sums the total")
    void countsEveryStatusAndSumsTotal() {
        when(permissionUtils.getUserId()).thenReturn("admin-uid");
        when(permissionUtils.isAdmin()).thenReturn(true);

        LocalDate from = LocalDate.of(2026, 3, 5);
        LocalDate to = LocalDate.of(2026, 3, 7);
        LocalDateTime expectedStart = from.atStartOfDay();
        LocalDateTime expectedEnd = to.plusDays(1).atStartOfDay();

        when(ticketRepository.countByStatusAndCreatedTimeBetween(eq(TicketStatus.OPEN), eq(expectedStart), eq(expectedEnd)))
                .thenReturn(3L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(eq(TicketStatus.IN_PROGRESS), eq(expectedStart), eq(expectedEnd)))
                .thenReturn(2L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(eq(TicketStatus.WAITING_FOR_CUSTOMER), eq(expectedStart), eq(expectedEnd)))
                .thenReturn(0L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(eq(TicketStatus.RESOLVED), eq(expectedStart), eq(expectedEnd)))
                .thenReturn(5L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(eq(TicketStatus.CLOSED), eq(expectedStart), eq(expectedEnd)))
                .thenReturn(1L);

        TicketStatsDtoOut result = service.getTicketStats(from, to);

        assertThat(result.getCountsByStatus())
                .hasSize(TicketStatus.values().length)
                .containsEntry(TicketStatus.OPEN, 3L)
                .containsEntry(TicketStatus.IN_PROGRESS, 2L)
                .containsEntry(TicketStatus.WAITING_FOR_CUSTOMER, 0L)
                .containsEntry(TicketStatus.RESOLVED, 5L)
                .containsEntry(TicketStatus.CLOSED, 1L);
        assertThat(result.getTotal()).isEqualTo(11L);

        for (TicketStatus status : TicketStatus.values()) {
            verify(ticketRepository).countByStatusAndCreatedTimeBetween(status, expectedStart, expectedEnd);
        }
    }

    @Test
    @DisplayName("accepts a range where from equals to")
    void acceptsSameDayRange() {
        when(permissionUtils.getUserId()).thenReturn("admin-uid");
        when(permissionUtils.isAdmin()).thenReturn(true);

        LocalDate day = LocalDate.of(2026, 4, 1);

        TicketStatsDtoOut result = service.getTicketStats(day, day);

        assertThat(result.getTotal()).isEqualTo(0L);
        verify(ticketRepository).countByStatusAndCreatedTimeBetween(
                eq(TicketStatus.OPEN), eq(day.atStartOfDay()), eq(day.plusDays(1).atStartOfDay()));
    }
}
