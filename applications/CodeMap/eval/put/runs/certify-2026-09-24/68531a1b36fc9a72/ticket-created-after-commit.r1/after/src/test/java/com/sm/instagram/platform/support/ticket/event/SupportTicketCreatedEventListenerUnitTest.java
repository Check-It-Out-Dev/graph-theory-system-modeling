package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.common.EmailService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketCreatedEventListener Unit Tests")
class SupportTicketCreatedEventListenerUnitTest {

    @Mock
    private EmailService emailService;

    private SupportTicketCreatedEventListener listener;

    @Test
    @DisplayName("should send the confirmation email with the event's data")
    void shouldSendConfirmationEmailWithEventData() {
        listener = new SupportTicketCreatedEventListener(emailService);

        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 42L, "customer@example.com", "TKT-12345", "Cannot log in", "pl", "magic-token-xyz");

        listener.onTicketCreated(event);

        verify(emailService).sendTicketCreationConfirmation(
                "customer@example.com", "TKT-12345", "Cannot log in", "pl", "magic-token-xyz");
    }

    @Test
    @DisplayName("should catch and log a failure from EmailService instead of rethrowing")
    void shouldCatchAndLogEmailServiceFailure() {
        listener = new SupportTicketCreatedEventListener(emailService);

        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 7L, "other@example.com", "TKT-99999", "Billing issue", "en", "magic-token-abc");

        doThrow(new RuntimeException("SMTP down"))
                .when(emailService).sendTicketCreationConfirmation(
                        "other@example.com", "TKT-99999", "Billing issue", "en", "magic-token-abc");

        assertDoesNotThrow(() -> listener.onTicketCreated(event));

        verify(emailService).sendTicketCreationConfirmation(
                "other@example.com", "TKT-99999", "Billing issue", "en", "magic-token-abc");
    }
}
