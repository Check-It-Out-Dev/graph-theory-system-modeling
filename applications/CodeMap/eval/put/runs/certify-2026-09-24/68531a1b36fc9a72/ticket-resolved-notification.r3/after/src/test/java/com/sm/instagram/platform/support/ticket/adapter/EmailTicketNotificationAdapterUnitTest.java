package com.sm.instagram.platform.support.ticket.adapter;

import com.sm.instagram.platform.support.common.EmailService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class EmailTicketNotificationAdapterUnitTest {

    @Mock
    private EmailService emailService;

    private EmailTicketNotificationAdapter adapter;

    @Test
    @DisplayName("should send an e-mail whose subject contains the ticket reference")
    void shouldSendEmailWithTicketReferenceInSubject() {
        adapter = new EmailTicketNotificationAdapter(emailService);

        adapter.notifyResolved("customer@example.com", "TKT-98765");

        ArgumentCaptor<String> toCaptor = ArgumentCaptor.forClass(String.class);
        ArgumentCaptor<String> subjectCaptor = ArgumentCaptor.forClass(String.class);
        ArgumentCaptor<String> textCaptor = ArgumentCaptor.forClass(String.class);
        verify(emailService).sendEmail(toCaptor.capture(), subjectCaptor.capture(), textCaptor.capture());

        assertThat(toCaptor.getValue()).isEqualTo("customer@example.com");
        assertThat(subjectCaptor.getValue()).contains("TKT-98765");
        assertThat(textCaptor.getValue()).contains("TKT-98765");
    }
}
