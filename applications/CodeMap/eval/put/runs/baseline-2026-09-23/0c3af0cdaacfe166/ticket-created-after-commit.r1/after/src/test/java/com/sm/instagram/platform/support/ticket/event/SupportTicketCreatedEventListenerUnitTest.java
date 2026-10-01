package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.common.EmailService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.mockito.Mockito.*;

/**
 * Unit test for {@link SupportTicketCreatedEventListener} — confirms the
 * confirmation e-mail is sent with the event's data, and that a send failure
 * is swallowed rather than rethrown (the transaction it follows already committed).
 */
@ExtendWith(MockitoExtension.class)
class SupportTicketCreatedEventListenerUnitTest {

    @Mock
    private EmailService emailService;

    @InjectMocks
    private SupportTicketCreatedEventListener listener;

    private SupportTicketCreatedEvent event;

    @BeforeEach
    void setUp() {
        event = new SupportTicketCreatedEvent(
                this, 42L, "customer@example.com", "REF-123", "Help needed", "en", "status-token");
    }

    @Test
    void sendsTheConfirmationWithTheEventData() {
        listener.onTicketCreated(event);

        verify(emailService).sendTicketCreationConfirmation(
                "customer@example.com", "REF-123", "Help needed", "en", "status-token");
    }

    @Test
    void swallowsAFailureFromTheMailServer() {
        doThrow(new RuntimeException("mail server down"))
                .when(emailService).sendTicketCreationConfirmation(any(), any(), any(), any(), any());

        listener.onTicketCreated(event);

        verify(emailService).sendTicketCreationConfirmation(
                "customer@example.com", "REF-123", "Help needed", "en", "status-token");
    }
}
