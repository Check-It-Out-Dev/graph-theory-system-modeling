package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.event.SupportTicketCreatedEvent;
import com.sm.instagram.platform.support.ticket.event.SupportTicketCreatedEventListener;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class SupportTicketCreatedEventListenerUnitTest {

    @Mock
    private EmailService emailService;

    @InjectMocks
    private SupportTicketCreatedEventListener listener;

    private final SupportTicketCreatedEvent event =
            new SupportTicketCreatedEvent(this, 1L, "user@example.com", "CIO-REF", "Login problem", "en", "token");

    @Test
    void sendsTheConfirmationWithTheEventData() {
        listener.onTicketCreated(event);
        verify(emailService).sendTicketCreationConfirmation("user@example.com", "CIO-REF", "Login problem", "en", "token");
    }

    @Test
    void logsAndSwallowsMailFailures() {
        doThrow(new IllegalStateException("smtp")).when(emailService)
                .sendTicketCreationConfirmation(anyString(), anyString(), anyString(), anyString(), anyString());
        assertThatCode(() -> listener.onTicketCreated(event)).doesNotThrowAnyException();
    }
}
