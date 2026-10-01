package com.sm.instagram.platform.unit.controller;

import com.sm.instagram.platform.common.exceptions.AuthenticationTranslatableException;
import com.sm.instagram.platform.support.ticket.SupportTicketController;
import com.sm.instagram.platform.support.ticket.dtos.TicketStatsDtoOut;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import jakarta.servlet.http.HttpServletRequest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
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
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link SupportTicketController#getTicketStats(LocalDate, LocalDate)}.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketController#getTicketStats Unit Tests")
class SupportTicketControllerStatsUnitTest {

    @Mock
    private SupportTicketService ticketService;

    @Mock
    private HttpServletRequest httpServletRequest;

    @Mock
    private SecurityContext securityContext;

    @Mock
    private Authentication authentication;

    @InjectMocks
    private SupportTicketController controller;

    private static final String TEST_FIREBASE_UID = "admin-firebase-uid";
    private static final LocalDate FROM = LocalDate.of(2026, 2, 1);
    private static final LocalDate TO = LocalDate.of(2026, 2, 28);

    @BeforeEach
    void setUp() {
        when(securityContext.getAuthentication()).thenReturn(authentication);
        SecurityContextHolder.setContext(securityContext);
    }

    @Test
    @DisplayName("returns the stats computed by the service for the requested range")
    void returnsTheStatsComputedByTheServiceForTheRequestedRange() {
        when(authentication.getPrincipal()).thenReturn(TEST_FIREBASE_UID);
        Map<TicketStatus, Long> counts = new EnumMap<>(TicketStatus.class);
        for (TicketStatus status : TicketStatus.values()) {
            counts.put(status, 0L);
        }
        counts.put(TicketStatus.OPEN, 7L);
        TicketStatsDtoOut expected = new TicketStatsDtoOut(counts, 7L);

        when(ticketService.getTicketStats(FROM, TO)).thenReturn(expected);

        ResponseEntity<TicketStatsDtoOut> response = controller.getTicketStats(FROM, TO);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).isSameAs(expected);
        assertThat(response.getBody().getTotal()).isEqualTo(7L);
        verify(ticketService).getTicketStats(FROM, TO);
    }

    @Test
    @DisplayName("throws when authentication is missing from the security context")
    void throwsWhenAuthenticationIsMissing() {
        when(securityContext.getAuthentication()).thenReturn(null);

        assertThatThrownBy(() -> controller.getTicketStats(FROM, TO))
                .isInstanceOf(AuthenticationTranslatableException.class);
    }

    @Test
    @DisplayName("throws when the principal is missing from the authentication")
    void throwsWhenThePrincipalIsMissing() {
        when(authentication.getPrincipal()).thenReturn(null);

        assertThatThrownBy(() -> controller.getTicketStats(FROM, TO))
                .isInstanceOf(AuthenticationTranslatableException.class);
    }
}
