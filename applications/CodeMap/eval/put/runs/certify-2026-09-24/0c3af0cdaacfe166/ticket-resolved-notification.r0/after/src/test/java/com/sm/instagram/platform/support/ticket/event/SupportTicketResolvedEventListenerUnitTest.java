package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class SupportTicketResolvedEventListenerUnitTest {

    @Mock
    private TicketNotificationPort ticketNotificationPort;

    @InjectMocks
    private SupportTicketResolvedEventListener listener;

    @Test
    void notifiesThePortWithTheContactEmailAndReference() {
        SupportTicketResolvedEvent event = new SupportTicketResolvedEvent(this, 42L, "customer@example.com", "TKT-42");

        listener.onTicketResolved(event);

        verify(ticketNotificationPort).notifyResolved("customer@example.com", "TKT-42");
    }

    @Test
    void doesNotPropagateAFailureFromThePort() {
        SupportTicketResolvedEvent event = new SupportTicketResolvedEvent(this, 42L, "customer@example.com", "TKT-42");
        doThrow(new RuntimeException("provider down"))
                .when(ticketNotificationPort).notifyResolved("customer@example.com", "TKT-42");

        listener.onTicketResolved(event);

        verify(ticketNotificationPort).notifyResolved("customer@example.com", "TKT-42");
    }
}
