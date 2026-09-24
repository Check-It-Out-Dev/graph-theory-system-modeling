package com.sm.instagram.platform.support.ticket.event;

import lombok.Getter;
import org.springframework.context.ApplicationEvent;

/**
 * Published by SupportTicketService#createTicket once the ticket is saved.
 * Handled by SupportTicketCreatedEventListener AFTER_COMMIT, so the confirmation e-mail
 * only goes out for a ticket that exists in the database.
 */
@Getter
public class SupportTicketCreatedEvent extends ApplicationEvent {

    private final Long ticketId;
    private final String contactEmail;
    private final String ticketReference;
    private final String subject;
    private final String language;
    private final String statusToken;

    public SupportTicketCreatedEvent(Object source, Long ticketId, String contactEmail, String ticketReference,
                                     String subject, String language, String statusToken) {
        super(source);
        this.ticketId = ticketId;
        this.contactEmail = contactEmail;
        this.ticketReference = ticketReference;
        this.subject = subject;
        this.language = language;
        this.statusToken = statusToken;
    }
}
