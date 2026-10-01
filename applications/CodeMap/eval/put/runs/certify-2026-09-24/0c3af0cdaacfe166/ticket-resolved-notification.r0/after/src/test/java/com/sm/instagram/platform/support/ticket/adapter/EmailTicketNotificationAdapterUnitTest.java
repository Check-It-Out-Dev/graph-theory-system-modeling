package com.sm.instagram.platform.support.ticket.adapter;

import com.sm.instagram.platform.support.common.EmailService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.mockito.ArgumentMatchers.contains;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class EmailTicketNotificationAdapterUnitTest {

    @Mock
    private EmailService emailService;

    @InjectMocks
    private EmailTicketNotificationAdapter adapter;

    @Test
    void sendsAnEmailWhoseSubjectContainsTheTicketReference() {
        adapter.notifyResolved("customer@example.com", "TKT-42");

        verify(emailService).sendEmail(eq("customer@example.com"), contains("TKT-42"), contains("TKT-42"));
    }
}
