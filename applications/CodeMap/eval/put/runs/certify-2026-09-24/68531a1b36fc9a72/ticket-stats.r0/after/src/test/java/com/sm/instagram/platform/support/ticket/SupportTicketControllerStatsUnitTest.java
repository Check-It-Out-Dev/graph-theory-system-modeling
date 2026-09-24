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
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

import java.time.LocalDate;
import java.util.EnumMap;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
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

    @InjectMocks
    private SupportTicketController controller;

    @BeforeEach
    void setUp() {
        SecurityContextHolder.setContext(securityContext);
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    @Test
    @DisplayName("returns the stats the service computes")
    void returnsStatsFromService() {
        when(securityContext.getAuthentication()).thenReturn(authentication);
        when(authentication.getName()).thenReturn("admin-uid");
        when(authentication.getPrincipal()).thenReturn("admin-uid");

        LocalDate from = LocalDate.of(2026, 1, 1);
        LocalDate to = LocalDate.of(2026, 1, 31);

        EnumMap<TicketStatus, Long> counts = new EnumMap<>(TicketStatus.class);
        counts.put(TicketStatus.OPEN, 4L);
        counts.put(TicketStatus.IN_PROGRESS, 1L);
        counts.put(TicketStatus.WAITING_FOR_CUSTOMER, 0L);
        counts.put(TicketStatus.RESOLVED, 2L);
        counts.put(TicketStatus.CLOSED, 3L);
        TicketStatsDtoOut expected = new TicketStatsDtoOut(counts, 10L);

        when(ticketService.getTicketStats(eq(from), eq(to))).thenReturn(expected);

        ResponseEntity<TicketStatsDtoOut> response = controller.getTicketStats(from, to);

        assertThat(response.getBody()).isSameAs(expected);
        assertThat(response.getBody().getTotal()).isEqualTo(10L);
        verify(ticketService).getTicketStats(from, to);
    }

    @Test
    @DisplayName("throws when the security context carries no authentication")
    void throwsWhenNotAuthenticated() {
        when(securityContext.getAuthentication()).thenReturn(null);

        assertThatThrownBy(() -> controller.getTicketStats(LocalDate.of(2026, 1, 1), LocalDate.of(2026, 1, 31)))
                .isInstanceOf(AuthenticationTranslatableException.class)
                .hasFieldOrPropertyWithValue("messageKey", "error.auth.not_authenticated");
    }
}
