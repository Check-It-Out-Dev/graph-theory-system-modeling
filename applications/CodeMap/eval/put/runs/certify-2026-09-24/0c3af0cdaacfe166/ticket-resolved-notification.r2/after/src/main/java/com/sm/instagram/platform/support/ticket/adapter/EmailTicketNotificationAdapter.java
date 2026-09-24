package com.sm.instagram.platform.support.ticket.adapter;

import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/**
 * E-mail implementation of {@link TicketNotificationPort}.
 * Primary implementation today; the team plans to move to another provider,
 * which is why callers depend on the port, not on this adapter.
 */
@Component
@RequiredArgsConstructor
public class EmailTicketNotificationAdapter implements TicketNotificationPort {

    private final EmailService emailService;

    @Override
    public void notifyResolved(String contactEmail, String ticketReference) {
        String subject = "Your support ticket " + ticketReference + " has been resolved";
        String text = "Your support ticket " + ticketReference + " has been marked as resolved. "
                + "If you still need help, simply reply to your ticket and it will be reopened.";
        emailService.sendEmail(contactEmail, subject, text);
    }
}
