package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.common.EmailService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicketCreatedEventListener Unit Tests")
class SupportTicketCreatedEventListenerUnitTest {

    @Mock
    private EmailService emailService;

    private SupportTicketCreatedEventListener listener;

    @BeforeEach
    void setUp() {
        listener = new SupportTicketCreatedEventListener(emailService);
    }

    @Test
    @DisplayName("sends the confirmation email with the event's data")
    void sendsConfirmationEmailWithEventData() {
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 42L, "reporter@example.com", "CIO-12345", "Cannot log in", "pl", "magic-token");

        listener.onTicketCreated(event);

        verify(emailService).sendTicketCreationConfirmation(
                eq("reporter@example.com"),
                eq("CIO-12345"),
                eq("Cannot log in"),
                eq("pl"),
                eq("magic-token")
        );
    }

    @Test
    @DisplayName("catches and logs a failure from the email service instead of rethrowing")
    void catchesAndLogsEmailFailure() {
        SupportTicketCreatedEvent event = new SupportTicketCreatedEvent(
                this, 42L, "reporter@example.com", "CIO-12345", "Cannot log in", "pl", "magic-token");
        doThrow(new RuntimeException("mail server down"))
                .when(emailService).sendTicketCreationConfirmation(
                        eq("reporter@example.com"), eq("CIO-12345"), eq("Cannot log in"), eq("pl"), eq("magic-token"));

        assertThatCode(() -> listener.onTicketCreated(event)).doesNotThrowAnyException();

        verify(emailService).sendTicketCreationConfirmation(
                eq("reporter@example.com"),
                eq("CIO-12345"),
                eq("Cannot log in"),
                eq("pl"),
                eq("magic-token")
        );
    }
}
