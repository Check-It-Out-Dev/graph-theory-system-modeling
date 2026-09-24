package com.sm.instagram.platform.support.ticket.adapter;

import com.sm.instagram.platform.support.common.EmailService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.context.MessageSource;

import java.util.Locale;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@DisplayName("EmailTicketNotificationAdapter Unit Tests")
class EmailTicketNotificationAdapterUnitTest {

    @Mock
    private EmailService emailService;

    @Mock
    private MessageSource messageSource;

    @Test
    @DisplayName("should send an email whose subject and body carry the ticket reference")
    void shouldSendEmailWithTicketReference() {
        EmailTicketNotificationAdapter adapter = new EmailTicketNotificationAdapter(emailService, messageSource);

        when(messageSource.getMessage(eq("email.ticket_resolved.subject"), any(), eq(Locale.ENGLISH)))
                .thenReturn("Your support ticket [TKT-99] has been resolved");
        when(messageSource.getMessage(eq("email.ticket_resolved.body"), any(), eq(Locale.ENGLISH)))
                .thenReturn("Your ticket TKT-99 has been resolved.");

        adapter.notifyResolved("customer@example.com", "TKT-99");

        ArgumentCaptor<String> toCaptor = ArgumentCaptor.forClass(String.class);
        ArgumentCaptor<String> subjectCaptor = ArgumentCaptor.forClass(String.class);
        ArgumentCaptor<String> textCaptor = ArgumentCaptor.forClass(String.class);
        verify(emailService).sendEmail(toCaptor.capture(), subjectCaptor.capture(), textCaptor.capture());

        assertThat(toCaptor.getValue()).isEqualTo("customer@example.com");
        assertThat(subjectCaptor.getValue()).contains("TKT-99");
        assertThat(textCaptor.getValue()).contains("TKT-99");
    }
}
