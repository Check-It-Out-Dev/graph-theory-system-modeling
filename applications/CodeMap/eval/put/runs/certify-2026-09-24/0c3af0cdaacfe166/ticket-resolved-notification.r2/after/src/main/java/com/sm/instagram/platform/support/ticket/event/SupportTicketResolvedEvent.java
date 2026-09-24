package com.sm.instagram.platform.support.ticket.event;

import lombok.Getter;
import org.springframework.context.ApplicationEvent;

/**
 * Published when a support ticket transitions to RESOLVED.
 * Handled by SupportTicketResolvedEventListener AFTER_COMMIT — so the status change
 * is durable before the customer is notified.
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
