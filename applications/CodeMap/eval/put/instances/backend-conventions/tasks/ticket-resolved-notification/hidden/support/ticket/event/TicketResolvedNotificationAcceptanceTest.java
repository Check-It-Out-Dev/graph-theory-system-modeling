package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.adapter.EmailTicketNotificationAdapter;
import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import com.sm.instagram.platform.support.ticket.models.TicketCategory;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import com.sm.instagram.platform.support.ticket.repositories.SupportTicketRepository;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.invocation.Invocation;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.context.ApplicationEventPublisher;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.contains;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Hidden acceptance test of the task ticket-resolved-notification (prompt under test, instance backend-conventions). */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class TicketResolvedNotificationAcceptanceTest {

    @Mock SupportTicketRepository ticketRepository;
    @Mock PermissionUtils permissionUtils;
    @Mock ApplicationEventPublisher eventPublisher;
    @Mock EmailService emailService;
    @Mock TicketNotificationPort port;
    @InjectMocks SupportTicketService service;
    @InjectMocks SupportTicketResolvedEventListener listener;
    @InjectMocks EmailTicketNotificationAdapter adapter;

    private SupportTicket ticket;

    @BeforeEach
    void setUp() {
        ticket = new SupportTicket();
        ticket.setId(42L);
        ticket.setContactEmail("customer@example.com");
        ticket.setTicketReference("CIO-20260923-RESOLVED");
        ticket.setSubject("Payout missing");
        ticket.setDescription("d");
        ticket.setCategory(TicketCategory.values()[0]);
        ticket.setStatus(TicketStatus.IN_PROGRESS);
        when(permissionUtils.isAdmin()).thenReturn(true);
        when(permissionUtils.getUserId()).thenReturn("admin-uid");
        when(ticketRepository.findById(42L)).thenReturn(Optional.of(ticket));
        when(ticketRepository.save(any(SupportTicket.class))).thenAnswer(inv -> inv.getArgument(0));
    }

    @Test
    void resolvingATicketPublishesTheEvent() {
        service.updateTicketStatus(42L, TicketStatus.RESOLVED);
        List<SupportTicketResolvedEvent> events = published();
        assertThat(events).hasSize(1);
        assertThat(events.get(0).getTicketId()).isEqualTo(42L);
        assertThat(events.get(0).getContactEmail()).isEqualTo("customer@example.com");
        assertThat(events.get(0).getTicketReference()).isEqualTo("CIO-20260923-RESOLVED");
        assertThat(Mockito.mockingDetails(emailService).getInvocations()).isEmpty();
    }

    @Test
    void anotherStatusPublishesNothing() {
        service.updateTicketStatus(42L, TicketStatus.WAITING_FOR_CUSTOMER);
        assertThat(published()).isEmpty();
    }

    @Test
    void theListenerNotifiesThroughThePort() {
        listener.onTicketResolved(new SupportTicketResolvedEvent(this, 42L, "customer@example.com", "CIO-7"));
        verify(port).notifyResolved("customer@example.com", "CIO-7");
    }

    @Test
    void aPortFailureDoesNotLeaveTheListener() {
        doThrow(new IllegalStateException("provider down")).when(port).notifyResolved(anyString(), anyString());
        assertThatCode(() -> listener.onTicketResolved(new SupportTicketResolvedEvent(this, 42L, "c@example.com", "CIO-7")))
                .doesNotThrowAnyException();
    }

    @Test
    void theEmailAdapterImplementsThePortAndNamesTheTicket() {
        assertThat(TicketNotificationPort.class.isInterface()).isTrue();
        assertThat(TicketNotificationPort.class.isAssignableFrom(EmailTicketNotificationAdapter.class)).isTrue();
        adapter.notifyResolved("customer@example.com", "CIO-7");
        verify(emailService).sendEmail(eq("customer@example.com"), contains("CIO-7"), anyString());
    }

    private List<SupportTicketResolvedEvent> published() {
        List<SupportTicketResolvedEvent> out = new ArrayList<>();
        for (Invocation i : Mockito.mockingDetails(eventPublisher).getInvocations()) {
            if (i.getMethod().getName().equals("publishEvent") && i.getArgument(0) instanceof SupportTicketResolvedEvent e) {
                out.add(e);
            }
        }
        return out;
    }
}
