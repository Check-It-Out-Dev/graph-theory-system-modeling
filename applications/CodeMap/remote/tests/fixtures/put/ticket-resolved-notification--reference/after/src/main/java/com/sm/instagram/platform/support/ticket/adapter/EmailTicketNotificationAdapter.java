package com.sm.instagram.platform.support.ticket.adapter;

import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/**
 * TicketNotificationPort over e-mail.
 */
@Component
@RequiredArgsConstructor
public class EmailTicketNotificationAdapter implements TicketNotificationPort {

    private final EmailService emailService;

    @Override
    public void notifyResolved(String contactEmail, String ticketReference) {
        emailService.sendEmail(
                contactEmail,
                "Your support ticket " + ticketReference + " is resolved",
                "Your support ticket " + ticketReference + " has been resolved. "
                        + "If the problem is still there, reply to the ticket and we will pick it up again.");
    }
}
