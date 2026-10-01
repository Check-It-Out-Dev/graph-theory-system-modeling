package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.common.EmailService;
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
                this, 42L, "reporter@example.com", "TKT-55555", "Help", "en", "magic-tok");

        listener.onTicketCreated(event);

        verify(emailService).sendTicketCreationConfirmation(
                "reporter@example.com", "TKT-55555", "Help", "en", "magic-tok");
    }

    @Test
    @DisplayName("should swallow email failures instead of rethrowing")
    void shouldSwallowEmailFailuresInsteadOfRethrowing() {
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 42L, "reporter@example.com", "TKT-55555", "Help", "en", "magic-tok");
        doThrow(new RuntimeException("mail server down"))
                .when(emailService).sendTicketCreationConfirmation(
                        "reporter@example.com", "TKT-55555", "Help", "en", "magic-tok");

        listener.onTicketCreated(event);
    }
}
