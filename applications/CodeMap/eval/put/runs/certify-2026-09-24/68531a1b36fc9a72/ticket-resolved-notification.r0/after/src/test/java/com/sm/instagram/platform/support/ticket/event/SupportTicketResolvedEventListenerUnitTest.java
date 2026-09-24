package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketResolvedEventListener Unit Tests")
class SupportTicketResolvedEventListenerUnitTest {

    @Mock
    private TicketNotificationPort ticketNotificationPort;

    @Test
    @DisplayName("should notify through the port with the event's contact email and reference")
    void shouldNotifyThroughThePort() {
        SupportTicketResolvedEventListener listener = new SupportTicketResolvedEventListener(ticketNotificationPort);
        SupportTicketResolvedEvent event = new SupportTicketResolvedEvent(
                this, 42L, "customer@example.com", "TKT-42");

        listener.onTicketResolved(event);

        verify(ticketNotificationPort).notifyResolved("customer@example.com", "TKT-42");
    }

    @Test
    @DisplayName("should log and swallow a port failure instead of throwing")
    void shouldSwallowPortFailure() {
        SupportTicketResolvedEventListener listener = new SupportTicketResolvedEventListener(ticketNotificationPort);
        SupportTicketResolvedEvent event = new SupportTicketResolvedEvent(
                this, 7L, "other@example.com", "TKT-7");
        doThrow(new RuntimeException("provider down"))
                .when(ticketNotificationPort).notifyResolved("other@example.com", "TKT-7");

        assertThatCode(() -> listener.onTicketResolved(event)).doesNotThrowAnyException();

        verify(ticketNotificationPort).notifyResolved("other@example.com", "TKT-7");
    }
}
