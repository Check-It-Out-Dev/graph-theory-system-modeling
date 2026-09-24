package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.ticket.event.SupportTicketResolvedEvent;
import com.sm.instagram.platform.support.ticket.event.SupportTicketResolvedEventListener;
import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketResolvedEventListener Unit Tests")
class SupportTicketResolvedEventListenerUnitTest {

    @Mock
    private TicketNotificationPort ticketNotificationPort;

    @InjectMocks
    private SupportTicketResolvedEventListener listener;

    @Test
    @DisplayName("should notify the customer through the port")
    void shouldNotifyCustomerThroughPort() {
        // Given
        SupportTicketResolvedEvent event = new SupportTicketResolvedEvent(this, 1L, "test@example.com", "TKT-12345");

        // When
        listener.onTicketResolved(event);

        // Then
        verify(ticketNotificationPort).notifyResolved("test@example.com", "TKT-12345");
    }

    @Test
    @DisplayName("should log and not throw when the port fails")
    void shouldNotThrowWhenPortFails() {
        // Given
        SupportTicketResolvedEvent event = new SupportTicketResolvedEvent(this, 1L, "test@example.com", "TKT-12345");
        doThrow(new RuntimeException("provider down"))
                .when(ticketNotificationPort).notifyResolved("test@example.com", "TKT-12345");

        // When/Then - must not propagate, the transaction already committed
        listener.onTicketResolved(event);
    }
}
