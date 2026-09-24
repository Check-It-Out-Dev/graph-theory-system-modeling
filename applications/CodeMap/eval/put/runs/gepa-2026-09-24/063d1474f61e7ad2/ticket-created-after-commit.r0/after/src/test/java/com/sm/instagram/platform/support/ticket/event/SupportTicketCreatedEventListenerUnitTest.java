package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.common.EmailService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

/**
 * Unit test for {@link SupportTicketCreatedEventListener} — the AFTER_COMMIT
 * handler that sends the ticket confirmation e-mail once the ticket is durable.
 */
@ExtendWith(MockitoExtension.class)
class SupportTicketCreatedEventListenerUnitTest {

    @Mock
    private EmailService emailService;

    @Test
    void sendsTheConfirmationEmailWithTheEventsData() {
        SupportTicketCreatedEventListener listener = new SupportTicketCreatedEventListener(emailService);
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 42L, "reporter@example.com", "TKT-12345", "Cannot log in", "en", "magic-tok");

        listener.onTicketCreated(event);

        verify(emailService).sendTicketCreationConfirmation(
                eq("reporter@example.com"),
                eq("TKT-12345"),
                eq("Cannot log in"),
                eq("en"),
                eq("magic-tok")
        );
    }

    @Test
    void swallowsAFailureFromEmailServiceInsteadOfPropagatingIt() {
        SupportTicketCreatedEventListener listener = new SupportTicketCreatedEventListener(emailService);
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 42L, "reporter@example.com", "TKT-12345", "Cannot log in", "en", "magic-tok");
        doThrow(new RuntimeException("mail server down"))
                .when(emailService).sendTicketCreationConfirmation(
                        eq("reporter@example.com"), eq("TKT-12345"), eq("Cannot log in"), eq("en"), eq("magic-tok"));

        assertDoesNotThrow(() -> listener.onTicketCreated(event));
    }
}
