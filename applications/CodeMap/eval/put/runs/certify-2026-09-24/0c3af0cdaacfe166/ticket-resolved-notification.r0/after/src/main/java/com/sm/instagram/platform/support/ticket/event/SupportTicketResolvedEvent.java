package com.sm.instagram.platform.support.ticket.event;

import lombok.Getter;
import org.springframework.context.ApplicationEvent;

/**
 * Published by SupportTicketService when an admin moves a ticket to RESOLVED.
 */
@Getter
public class SupportTicketResolvedEvent extends ApplicationEvent {

    private final Long ticketId;
    private final String contactEmail;
    private final String ticketReference;

    public SupportTicketResolvedEvent(Object source, Long ticketId, String contactEmail, String ticketReference) {
        super(source);
        this.ticketId = ticketId;
        this.contactEmail = contactEmail;
        this.ticketReference = ticketReference;
    }
}
