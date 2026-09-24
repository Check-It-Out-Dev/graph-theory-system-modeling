package com.sm.instagram.platform.support.ticket.adapter;

import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/**
 * E-mail adapter for {@link TicketNotificationPort}, implemented on top of
 * {@link EmailService}.
 */
@Component
@RequiredArgsConstructor
public class EmailTicketNotificationAdapter implements TicketNotificationPort {

    private final EmailService emailService;

    @Override
    public void notifyResolved(String contactEmail, String ticketReference) {
        String subject = "Your support ticket [" + ticketReference + "] has been resolved";
        String text = "Your support ticket " + ticketReference
                + " has been marked as resolved. If you still need help, just reply to your ticket.";
        emailService.sendEmail(contactEmail, subject, text);
    }
}
