package com.sm.instagram.platform.support.ticket.services;

import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.repositories.SupportTicketRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.invocation.Invocation;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Hidden acceptance test of the task stale-ticket-close (prompt under test, instance backend-conventions). */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class StaleTicketCloseAcceptanceTest {

    @Mock SupportTicketService supportTicketService;
    @InjectMocks StaleTicketCloseCronJob job;

    @Mock SupportTicketRepository repository;
    @Mock EmailService emailService;
    @InjectMocks SupportTicketService service;

    @Test
    void theJobClosesWithTheConfiguredAge() {
        ReflectionTestUtils.setField(job, "enabled", true);
        ReflectionTestUtils.setField(job, "olderThanDays", 21);
        when(supportTicketService.closeStaleResolvedTickets(anyInt())).thenReturn(3);
        job.closeStaleTickets();
        verify(supportTicketService).closeStaleResolvedTickets(21);
    }

    @Test
    void aDisabledJobDoesNothing() {
        ReflectionTestUtils.setField(job, "enabled", false);
        ReflectionTestUtils.setField(job, "olderThanDays", 14);
        job.closeStaleTickets();
        verify(supportTicketService, never()).closeStaleResolvedTickets(anyInt());
    }

    @Test
    void aFailureIsNotThrownOutOfTheJob() {
        ReflectionTestUtils.setField(job, "enabled", true);
        ReflectionTestUtils.setField(job, "olderThanDays", 14);
        when(supportTicketService.closeStaleResolvedTickets(anyInt())).thenThrow(new IllegalStateException("db down"));
        assertThatCode(() -> job.closeStaleTickets()).doesNotThrowAnyException();
    }

    @Test
    void theServiceClosesResolvedTicketsOlderThanTheCutoff() {
        SupportTicket first = ticket(1L);
        SupportTicket second = ticket(2L);
        when(repository.findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), any(LocalDateTime.class)))
                .thenReturn(new ArrayList<>(List.of(first, second)));

        int closed = service.closeStaleResolvedTickets(14);

        assertThat(closed).isEqualTo(2);
        assertThat(first.getStatus()).isEqualTo(TicketStatus.CLOSED);
        assertThat(second.getStatus()).isEqualTo(TicketStatus.CLOSED);
        ArgumentCaptor<LocalDateTime> cutoff = ArgumentCaptor.forClass(LocalDateTime.class);
        verify(repository).findByStatusAndResolvedTimeBefore(eq(TicketStatus.RESOLVED), cutoff.capture());
        assertThat(Duration.between(cutoff.getValue(), LocalDateTime.now().minusDays(14)).abs()).isLessThan(Duration.ofMinutes(5));
        assertThat(saved(repository)).contains(first, second);
    }

    private static SupportTicket ticket(Long id) {
        SupportTicket t = new SupportTicket();
        t.setId(id);
        t.setStatus(TicketStatus.RESOLVED);
        t.setResolvedTime(LocalDateTime.now().minusDays(30));
        return t;
    }

    /** Whatever the service handed to save, saveAll or saveAndFlush. */
    private static List<Object> saved(Object repository) {
        List<Object> out = new ArrayList<>();
        for (Invocation i : Mockito.mockingDetails(repository).getInvocations()) {
            if (!i.getMethod().getName().startsWith("save")) continue;
            for (Object arg : i.getArguments()) {
                if (arg instanceof Collection<?> c) out.addAll(c);
                else if (arg instanceof Iterable<?> it) it.forEach(out::add);
                else out.add(arg);
            }
        }
        return out;
    }
}
