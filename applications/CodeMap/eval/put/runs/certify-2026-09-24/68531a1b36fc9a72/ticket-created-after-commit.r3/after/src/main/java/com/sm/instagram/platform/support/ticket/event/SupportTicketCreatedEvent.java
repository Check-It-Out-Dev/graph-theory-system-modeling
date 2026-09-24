package com.sm.instagram.platform.support.ticket.event;

import lombok.Getter;
import org.springframework.context.ApplicationEvent;

/**
 * Published by SupportTicketService after a ticket is created.
 * Handled by SupportTicketCreatedEventListener AFTER_COMMIT — so the ticket exists in DB
 * before the confirmation email (and its magic-link token) is sent.
 */
@Getter
public class SupportTicketCreatedEvent extends ApplicationEvent {

    private final Long ticketId;
    private final String contactEmail;
    private final String ticketReference;
    private final String subject;
    private final String language;
    private final String statusToken;

    public SupportTicketCreatedEvent(Object source, Long ticketId, String contactEmail,
                                      String ticketReference, String subject,
                                      String language, String statusToken) {
        super(source);
        this.ticketId = ticketId;
        this.contactEmail = contactEmail;
        this.ticketReference = ticketReference;
        this.subject = subject;
        this.language = language;
        this.statusToken = statusToken;
    }
}
