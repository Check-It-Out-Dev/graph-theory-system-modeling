package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.event.SupportTicketCreatedEvent;
import com.sm.instagram.platform.support.ticket.event.SupportTicketCreatedEventListener;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketCreatedEventListener Unit Tests")
class SupportTicketCreatedEventListenerUnitTest {

    @Mock
    private EmailService emailService;

    @InjectMocks
    private SupportTicketCreatedEventListener listener;

    @Test
    @DisplayName("should send confirmation email with the event's data")
    void shouldSendConfirmationEmailWithEventData() {
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 42L, "reporter@example.com", "TKT-99999", "Help", "en", "magic-tok");

        listener.onTicketCreated(event);

        verify(emailService).sendTicketCreationConfirmation(
                "reporter@example.com", "TKT-99999", "Help", "en", "magic-tok");
    }

    @Test
    @DisplayName("should catch and log a failure instead of rethrowing")
    void shouldCatchAndLogFailureInsteadOfRethrowing() {
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 42L, "reporter@example.com", "TKT-99999", "Help", "en", "magic-tok");
        doThrow(new RuntimeException("mail server down"))
                .when(emailService).sendTicketCreationConfirmation(
                        "reporter@example.com", "TKT-99999", "Help", "en", "magic-tok");

        listener.onTicketCreated(event);
    }
}
