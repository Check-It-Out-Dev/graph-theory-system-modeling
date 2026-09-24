package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.common.EmailService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

/**
 * Unit test for {@link SupportTicketCreatedEventListener} — confirms the
 * confirmation e-mail is sent with the event's data, and that a failure to
 * send does not propagate (the transaction has already committed).
 */
@ExtendWith(MockitoExtension.class)
class SupportTicketCreatedEventListenerUnitTest {

    @Mock
    private EmailService emailService;

    @InjectMocks
    private SupportTicketCreatedEventListener listener;

    @Test
    void sendsTheConfirmationWithTheEventData() {
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 42L, "customer@example.com", "TKT-00042", "Help needed", "en", "token-abc");

        listener.onTicketCreated(event);

        verify(emailService).sendTicketCreationConfirmation(
                "customer@example.com", "TKT-00042", "Help needed", "en", "token-abc");
    }

    @Test
    void aFailedSendIsCaughtAndNotRethrown() {
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 7L, "customer@example.com", "TKT-00007", "Broken", "pl", null);
        doThrow(new RuntimeException("mail server down"))
                .when(emailService).sendTicketCreationConfirmation(anyString(), anyString(), anyString(), anyString(), any());

        listener.onTicketCreated(event);
    }
}
