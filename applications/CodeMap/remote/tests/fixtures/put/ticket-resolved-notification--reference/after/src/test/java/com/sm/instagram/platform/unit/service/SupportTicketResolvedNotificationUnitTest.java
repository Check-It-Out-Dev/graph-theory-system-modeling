package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.adapter.EmailTicketNotificationAdapter;
import com.sm.instagram.platform.support.ticket.event.SupportTicketResolvedEvent;
import com.sm.instagram.platform.support.ticket.event.SupportTicketResolvedEventListener;
import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.contains;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class SupportTicketResolvedNotificationUnitTest {

    @Mock
    private TicketNotificationPort port;

    @Mock
    private EmailService emailService;

    @Test
    void listenerNotifiesThroughThePort() {
        new SupportTicketResolvedEventListener(port)
                .onTicketResolved(new SupportTicketResolvedEvent(this, 1L, "c@example.com", "CIO-9"));
        verify(port).notifyResolved("c@example.com", "CIO-9");
    }

    @Test
    void listenerSwallowsPortFailures() {
        doThrow(new IllegalStateException("down")).when(port).notifyResolved(anyString(), anyString());
        assertThatCode(() -> new SupportTicketResolvedEventListener(port)
                .onTicketResolved(new SupportTicketResolvedEvent(this, 1L, "c@example.com", "CIO-9")))
                .doesNotThrowAnyException();
    }

    @Test
    void emailAdapterPutsTheReferenceInTheSubject() {
        new EmailTicketNotificationAdapter(emailService).notifyResolved("c@example.com", "CIO-9");
        verify(emailService).sendEmail(eq("c@example.com"), contains("CIO-9"), anyString());
    }
}
