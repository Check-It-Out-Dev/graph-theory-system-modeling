package com.sm.instagram.platform.support.ticket.adapter;

import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.MessageSource;
import org.springframework.stereotype.Component;

import java.util.Locale;

/**
 * E-mail implementation of {@link TicketNotificationPort}.
 * Today's delivery channel for ticket-resolved notices; a future provider
 * swap only needs a new adapter behind this port.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class EmailTicketNotificationAdapter implements TicketNotificationPort {

    private final EmailService emailService;
    private final MessageSource messageSource;

    @Override
    public void notifyResolved(String contactEmail, String ticketReference) {
        String subject = messageSource.getMessage(
                "email.ticket_resolved.subject", new Object[]{ticketReference}, Locale.ENGLISH);
        String text = messageSource.getMessage(
                "email.ticket_resolved.body", new Object[]{ticketReference}, Locale.ENGLISH);

        emailService.sendEmail(contactEmail, subject, text);
    }
}
