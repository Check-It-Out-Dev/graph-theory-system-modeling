package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.adapter.EmailTicketNotificationAdapter;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.mockito.ArgumentMatchers.contains;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
@DisplayName("EmailTicketNotificationAdapter Unit Tests")
class EmailTicketNotificationAdapterUnitTest {

    @Mock
    private EmailService emailService;

    @InjectMocks
    private EmailTicketNotificationAdapter adapter;

    @Test
    @DisplayName("should send an email whose subject contains the ticket reference")
    void shouldSendEmailWithReferenceInSubject() {
        // When
        adapter.notifyResolved("test@example.com", "TKT-12345");

        // Then
        verify(emailService).sendEmail(eq("test@example.com"), contains("TKT-12345"), contains("TKT-12345"));
    }
}
