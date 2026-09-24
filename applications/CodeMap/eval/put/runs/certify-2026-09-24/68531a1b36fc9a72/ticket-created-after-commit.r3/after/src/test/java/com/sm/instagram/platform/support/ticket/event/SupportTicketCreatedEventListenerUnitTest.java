package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.common.EmailService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

/**
 * Unit test for {@link SupportTicketCreatedEventListener} — the AFTER_COMMIT
 * handler that sends the ticket-creation confirmation email.
 */
@ExtendWith(MockitoExtension.class)
class SupportTicketCreatedEventListenerUnitTest {

    @Mock
    private EmailService emailService;

    @Test
    void sendsTheConfirmationWithTheEventsValues() {
        SupportTicketCreatedEventListener listener = new SupportTicketCreatedEventListener(emailService);
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 12L, "reporter@example.com", "CIO-55555", "Cannot log in", "en", "magic-tok-7");

        listener.onTicketCreated(event);

        verify(emailService).sendTicketCreationConfirmation(
                "reporter@example.com", "CIO-55555", "Cannot log in", "en", "magic-tok-7");
    }

    @Test
    void swallowsAndLogsAFailureFromTheEmailService() {
        SupportTicketCreatedEventListener listener = new SupportTicketCreatedEventListener(emailService);
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 13L, "reporter2@example.com", "CIO-66666", "Billing issue", "pl", "magic-tok-8");
        doThrow(new RuntimeException("SMTP down"))
                .when(emailService)
                .sendTicketCreationConfirmation("reporter2@example.com", "CIO-66666", "Billing issue", "pl", "magic-tok-8");

        assertDoesNotThrow(() -> listener.onTicketCreated(event));

        verify(emailService).sendTicketCreationConfirmation(
                "reporter2@example.com", "CIO-66666", "Billing issue", "pl", "magic-tok-8");
    }
}
