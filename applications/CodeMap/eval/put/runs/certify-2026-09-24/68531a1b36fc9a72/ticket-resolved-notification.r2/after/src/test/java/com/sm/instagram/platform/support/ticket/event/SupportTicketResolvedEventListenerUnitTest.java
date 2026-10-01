package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class SupportTicketResolvedEventListenerUnitTest {

    @Mock
    private TicketNotificationPort ticketNotificationPort;

    @Test
    void onTicketResolvedNotifiesTheCustomerThroughThePort() {
        SupportTicketResolvedEventListener listener = new SupportTicketResolvedEventListener(ticketNotificationPort);
        SupportTicketResolvedEvent event = new SupportTicketResolvedEvent(
                this, 501L, "customer@example.com", "TKT-70021");

        listener.onTicketResolved(event);

        verify(ticketNotificationPort).notifyResolved("customer@example.com", "TKT-70021");
    }

    @Test
    void onTicketResolvedSwallowsAPortFailureInsteadOfThrowing() {
        SupportTicketResolvedEventListener listener = new SupportTicketResolvedEventListener(ticketNotificationPort);
        SupportTicketResolvedEvent event = new SupportTicketResolvedEvent(
                this, 502L, "other@example.com", "TKT-70022");
        doThrow(new RuntimeException("provider down"))
                .when(ticketNotificationPort).notifyResolved("other@example.com", "TKT-70022");

        assertDoesNotThrow(() -> listener.onTicketResolved(event));

        verify(ticketNotificationPort).notifyResolved("other@example.com", "TKT-70022");
    }
}
