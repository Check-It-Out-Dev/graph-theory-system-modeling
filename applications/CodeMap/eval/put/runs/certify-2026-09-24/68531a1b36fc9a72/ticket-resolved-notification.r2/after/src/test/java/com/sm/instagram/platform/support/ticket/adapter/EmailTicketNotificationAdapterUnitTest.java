package com.sm.instagram.platform.support.ticket.adapter;

import com.sm.instagram.platform.support.common.EmailService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class EmailTicketNotificationAdapterUnitTest {

    @Mock
    private EmailService emailService;

    @Test
    void notifyResolvedSendsAnEmailWithTheReferenceInTheSubject() {
        EmailTicketNotificationAdapter adapter = new EmailTicketNotificationAdapter(emailService);

        adapter.notifyResolved("customer@example.com", "TKT-88031");

        ArgumentCaptor<String> subjectCaptor = ArgumentCaptor.forClass(String.class);
        ArgumentCaptor<String> textCaptor = ArgumentCaptor.forClass(String.class);
        verify(emailService).sendEmail(eq("customer@example.com"), subjectCaptor.capture(), textCaptor.capture());

        assertThat(subjectCaptor.getValue()).contains("TKT-88031");
        assertThat(textCaptor.getValue()).contains("TKT-88031");
    }
}
