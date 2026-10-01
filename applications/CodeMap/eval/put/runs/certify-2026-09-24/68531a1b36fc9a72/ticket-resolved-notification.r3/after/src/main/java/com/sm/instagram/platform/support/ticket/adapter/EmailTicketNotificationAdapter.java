package com.sm.instagram.platform.support.ticket.adapter;

import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/**
 * E-mail implementation of {@link TicketNotificationPort}.
 * Primary implementation: SMTP via {@link EmailService}.
 */
@Component
@RequiredArgsConstructor
public class EmailTicketNotificationAdapter implements TicketNotificationPort {

    private final EmailService emailService;

    @Override
    public void notifyResolved(String contactEmail, String ticketReference) {
        String subject = "Your support ticket " + ticketReference + " has been resolved";
        String text = "<p>Your support ticket <strong>" + ticketReference
                + "</strong> has been resolved. If you have further questions, please reply to this ticket.</p>";
        emailService.sendEmail(contactEmail, subject, text);
    }
}
