package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import org.junit.jupiter.api.DisplayName;
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

    private SupportTicketResolvedEventListener listener;

    @Test
    @DisplayName("should notify the contact through the port with the event's e-mail and reference")
    void shouldNotifyContactThroughPort() {
        listener = new SupportTicketResolvedEventListener(ticketNotificationPort);
        SupportTicketResolvedEvent event =
                new SupportTicketResolvedEvent(this, 55L, "customer@example.com", "TKT-55555");

        listener.onTicketResolved(event);

        verify(ticketNotificationPort).notifyResolved("customer@example.com", "TKT-55555");
    }

    @Test
    @DisplayName("should log and swallow a failure of the notification port instead of rethrowing")
    void shouldSwallowPortFailure() {
        listener = new SupportTicketResolvedEventListener(ticketNotificationPort);
        SupportTicketResolvedEvent event =
                new SupportTicketResolvedEvent(this, 77L, "customer@example.com", "TKT-77777");
        doThrow(new RuntimeException("provider unreachable"))
                .when(ticketNotificationPort).notifyResolved("customer@example.com", "TKT-77777");

        assertDoesNotThrow(() -> listener.onTicketResolved(event));

        verify(ticketNotificationPort).notifyResolved("customer@example.com", "TKT-77777");
    }
}
