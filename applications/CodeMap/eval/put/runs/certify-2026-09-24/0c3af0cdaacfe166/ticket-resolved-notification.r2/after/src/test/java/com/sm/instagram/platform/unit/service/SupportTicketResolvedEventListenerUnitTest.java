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
    private TicketNotificationPort notificationPort;

    @InjectMocks
    private SupportTicketResolvedEventListener listener;

    @Test
    @DisplayName("should notify the customer through the port when a ticket is resolved")
    void shouldNotifyCustomerThroughPort() {
        // Given
        SupportTicketResolvedEvent event = new SupportTicketResolvedEvent(this, 1L, "test@example.com", "TKT-12345");

        // When
        listener.onTicketResolved(event);

        // Then
        verify(notificationPort).notifyResolved("test@example.com", "TKT-12345");
    }

    @Test
    @DisplayName("should log and not rethrow when the notification port fails")
    void shouldLogAndNotRethrowWhenPortFails() {
        // Given
        SupportTicketResolvedEvent event = new SupportTicketResolvedEvent(this, 1L, "test@example.com", "TKT-12345");
        doThrow(new RuntimeException("provider down"))
                .when(notificationPort).notifyResolved("test@example.com", "TKT-12345");

        // When/Then - must not throw
        listener.onTicketResolved(event);

        verify(notificationPort).notifyResolved("test@example.com", "TKT-12345");
    }
}
