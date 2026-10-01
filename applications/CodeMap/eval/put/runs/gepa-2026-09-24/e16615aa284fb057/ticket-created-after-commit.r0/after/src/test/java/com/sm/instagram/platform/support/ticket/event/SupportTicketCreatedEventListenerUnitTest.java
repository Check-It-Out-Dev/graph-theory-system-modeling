package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.common.EmailService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

/**
 * Unit tests for {@link SupportTicketCreatedEventListener} — the AFTER_COMMIT
 * handler that sends the ticket confirmation email once the ticket is durably saved.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketCreatedEventListener Unit Tests")
class SupportTicketCreatedEventListenerUnitTest {

    @Mock
    private EmailService emailService;

    @InjectMocks
    private SupportTicketCreatedEventListener listener;

    @Test
    @DisplayName("should send the confirmation email with the event's data")
    void shouldSendConfirmationEmailWithEventData() {
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 42L, "reporter@example.com", "TKT-99999", "Cannot log in", "pl", "status-tok-xyz");

        listener.onTicketCreated(event);

        verify(emailService).sendTicketCreationConfirmation(
                "reporter@example.com", "TKT-99999", "Cannot log in", "pl", "status-tok-xyz");
    }

    @Test
    @DisplayName("should catch and log a failure instead of rethrowing it")
    void shouldCatchAndLogFailureInsteadOfRethrowing() {
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 42L, "reporter@example.com", "TKT-99999", "Cannot log in", "pl", "status-tok-xyz");
        doThrow(new RuntimeException("mail server down"))
                .when(emailService)
                .sendTicketCreationConfirmation("reporter@example.com", "TKT-99999", "Cannot log in", "pl", "status-tok-xyz");

        assertDoesNotThrow(() -> listener.onTicketCreated(event));

        verify(emailService).sendTicketCreationConfirmation(
                "reporter@example.com", "TKT-99999", "Cannot log in", "pl", "status-tok-xyz");
    }
}
