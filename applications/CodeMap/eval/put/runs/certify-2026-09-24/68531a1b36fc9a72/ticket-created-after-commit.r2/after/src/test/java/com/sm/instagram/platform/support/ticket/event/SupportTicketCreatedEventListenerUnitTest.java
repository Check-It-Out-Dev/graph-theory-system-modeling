package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.common.EmailService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class SupportTicketCreatedEventListenerUnitTest {

    @Mock
    private EmailService emailService;

    private SupportTicketCreatedEventListener listener;

    @Test
    void sendsTheConfirmationWithTheEventsFields() {
        listener = new SupportTicketCreatedEventListener(emailService);
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 77L, "reporter@example.com", "TKT-77777", "Broken upload", "pl", "token-xyz");

        listener.onTicketCreated(event);

        verify(emailService).sendTicketCreationConfirmation(
                "reporter@example.com", "TKT-77777", "Broken upload", "pl", "token-xyz");
    }

    @Test
    void catchesAndLogsAFailureInsteadOfRethrowing() {
        listener = new SupportTicketCreatedEventListener(emailService);
        doThrow(new RuntimeException("SMTP down"))
                .when(emailService).sendTicketCreationConfirmation(
                        "reporter@example.com", "TKT-77777", "Broken upload", "pl", "token-xyz");
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 77L, "reporter@example.com", "TKT-77777", "Broken upload", "pl", "token-xyz");

        assertThatCode(() -> listener.onTicketCreated(event)).doesNotThrowAnyException();

        verify(emailService).sendTicketCreationConfirmation(
                "reporter@example.com", "TKT-77777", "Broken upload", "pl", "token-xyz");
    }
}
