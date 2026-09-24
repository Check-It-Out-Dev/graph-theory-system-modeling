package com.sm.instagram.platform.support.ticket;

import com.sm.instagram.platform.common.exceptions.AuthenticationTranslatableException;
import com.sm.instagram.platform.support.ticket.dtos.TicketStatsDtoOut;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

import java.time.LocalDate;
import java.util.EnumMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link SupportTicketController#getTicketStats(LocalDate, LocalDate)}.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketController#getTicketStats")
class SupportTicketControllerStatsUnitTest {

    @Mock
    private SupportTicketService ticketService;

    @Mock
    private SecurityContext securityContext;

    @Mock
    private Authentication authentication;

    private SupportTicketController controller;

    private static final String ADMIN_UID = "admin-firebase-uid";

    @BeforeEach
    void setUp() {
        controller = new SupportTicketController(ticketService);
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    @Test
    @DisplayName("returns the stats the service computes for an authenticated caller")
    void returnsTheStatsFromTheServiceForAnAuthenticatedCaller() {
        when(securityContext.getAuthentication()).thenReturn(authentication);
        when(authentication.getPrincipal()).thenReturn(ADMIN_UID);
        when(authentication.getName()).thenReturn(ADMIN_UID);
        SecurityContextHolder.setContext(securityContext);

        LocalDate from = LocalDate.of(2026, 1, 1);
        LocalDate to = LocalDate.of(2026, 1, 31);

        Map<TicketStatus, Long> counts = new EnumMap<>(TicketStatus.class);
        for (TicketStatus status : TicketStatus.values()) {
            counts.put(status, 0L);
        }
        counts.put(TicketStatus.OPEN, 4L);
        TicketStatsDtoOut stats = new TicketStatsDtoOut(counts, 4L);

        when(ticketService.getTicketStats(from, to)).thenReturn(stats);

        ResponseEntity<TicketStatsDtoOut> response = controller.getTicketStats(from, to);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).isNotNull();
        assertThat(response.getBody().getTotal()).isEqualTo(4L);
        assertThat(response.getBody().getCountsByStatus().get(TicketStatus.OPEN)).isEqualTo(4L);
        verify(ticketService).getTicketStats(from, to);
    }

    @Test
    @DisplayName("throws when authentication is missing")
    void throwsWhenAuthenticationIsMissing() {
        when(securityContext.getAuthentication()).thenReturn(null);
        SecurityContextHolder.setContext(securityContext);

        LocalDate from = LocalDate.of(2026, 1, 1);
        LocalDate to = LocalDate.of(2026, 1, 31);

        assertThatThrownBy(() -> controller.getTicketStats(from, to))
                .isInstanceOf(AuthenticationTranslatableException.class);
        verifyNoInteractions(ticketService);
    }
}
