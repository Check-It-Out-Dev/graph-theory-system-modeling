package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SupportTicketResolvedEventListenerUnitTest {

    @Mock
    private TicketNotificationPort ticketNotificationPort;

    private SupportTicketResolvedEventListener listener;

    @BeforeEach
    void setUp() {
        listener = new SupportTicketResolvedEventListener(ticketNotificationPort);
    }

    @Test
    @DisplayName("should notify the customer through the port")
    void shouldNotifyCustomerThroughThePort() {
        SupportTicketResolvedEvent event = new SupportTicketResolvedEvent(this, 42L, "customer@example.com", "TKT-99887");

        listener.onTicketResolved(event);

        verify(ticketNotificationPort).notifyResolved("customer@example.com", "TKT-99887");
    }

    @Test
    @DisplayName("should log and swallow a port failure instead of throwing")
    void shouldLogAndSwallowPortFailure() {
        SupportTicketResolvedEvent event = new SupportTicketResolvedEvent(this, 42L, "customer@example.com", "TKT-99887");
        doThrow(new RuntimeException("provider unavailable"))
                .when(ticketNotificationPort).notifyResolved(anyString(), anyString());

        assertThatCode(() -> listener.onTicketResolved(event)).doesNotThrowAnyException();

        verify(ticketNotificationPort).notifyResolved("customer@example.com", "TKT-99887");
    }
}
