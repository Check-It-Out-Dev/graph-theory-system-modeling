package com.sm.instagram.platform.support.ticket.services;

import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.common.exceptions.InsufficientPermissionsException;
import com.sm.instagram.platform.common.exceptions.TranslatableException;
import com.sm.instagram.platform.support.ticket.dtos.TicketStatsDtoOut;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.repositories.SupportTicketRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Properties;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.catchThrowable;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Hidden acceptance test of the task ticket-stats (prompt under test, instance backend-conventions). */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class TicketStatsAcceptanceTest {

    private static final String KEY = "error.validation.invalid_date_range";

    @Mock SupportTicketRepository ticketRepository;
    @Mock PermissionUtils permissionUtils;
    @InjectMocks SupportTicketService service;

    @Test
    void everyStatusIsCountedOverTheWholeLastDay() {
        when(permissionUtils.isAdmin()).thenReturn(true);
        when(permissionUtils.getUserId()).thenReturn("admin-uid");
        LocalDateTime start = LocalDate.of(2026, 9, 1).atStartOfDay();
        LocalDateTime end = LocalDate.of(2026, 9, 11).atStartOfDay();
        when(ticketRepository.countByStatusAndCreatedTimeBetween(any(), any(), any())).thenReturn(0L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.OPEN, start, end)).thenReturn(5L);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(TicketStatus.RESOLVED, start, end)).thenReturn(2L);

        TicketStatsDtoOut stats = service.getTicketStats(LocalDate.of(2026, 9, 1), LocalDate.of(2026, 9, 10));

        assertThat(stats.getCountsByStatus()).containsOnlyKeys(TicketStatus.values());
        assertThat(stats.getCountsByStatus().get(TicketStatus.OPEN)).isEqualTo(5L);
        assertThat(stats.getCountsByStatus().get(TicketStatus.RESOLVED)).isEqualTo(2L);
        assertThat(stats.getCountsByStatus().get(TicketStatus.CLOSED)).isZero();
        assertThat(stats.getTotal()).isEqualTo(7L);
        for (TicketStatus status : TicketStatus.values()) {
            verify(ticketRepository).countByStatusAndCreatedTimeBetween(eq(status), eq(start), eq(end));
        }
    }

    @Test
    void aReversedPeriodIsATranslatableError() {
        when(permissionUtils.isAdmin()).thenReturn(true);
        Throwable thrown = catchThrowable(() -> service.getTicketStats(LocalDate.of(2026, 9, 10), LocalDate.of(2026, 9, 1)));
        assertThat(thrown).isInstanceOf(TranslatableException.class);
        assertThat(((TranslatableException) thrown).getMessageKey()).isEqualTo(KEY);
    }

    @Test
    void aSingleDayIsAValidPeriod() {
        when(permissionUtils.isAdmin()).thenReturn(true);
        when(ticketRepository.countByStatusAndCreatedTimeBetween(any(), any(), any())).thenReturn(1L);
        TicketStatsDtoOut stats = service.getTicketStats(LocalDate.of(2026, 9, 5), LocalDate.of(2026, 9, 5));
        assertThat(stats.getTotal()).isEqualTo(TicketStatus.values().length);
    }

    @Test
    void aCallerWhoIsNotAnAdminIsRefused() {
        when(permissionUtils.isAdmin()).thenReturn(false);
        when(permissionUtils.getUserId()).thenReturn("user-uid");
        Throwable thrown = catchThrowable(() -> service.getTicketStats(LocalDate.of(2026, 9, 1), LocalDate.of(2026, 9, 2)));
        assertThat(thrown).isInstanceOf(InsufficientPermissionsException.class);
    }

    @Test
    void theKeyIsTranslatedInBothLanguages() throws IOException {
        assertThat(bundle("messages_en.properties").getProperty(KEY)).isNotBlank();
        assertThat(bundle("messages_pl.properties").getProperty(KEY)).isNotBlank();
    }

    private static Properties bundle(String name) throws IOException {
        Properties p = new Properties();
        try (InputStream in = TicketStatsAcceptanceTest.class.getClassLoader().getResourceAsStream(name)) {
            assertThat(in).as(name).isNotNull();
            p.load(new InputStreamReader(in, StandardCharsets.UTF_8));
        }
        return p;
    }
}
