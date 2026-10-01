package com.sm.instagram.platform.support.ticket.port;

/**
 * Port for telling a customer about their support ticket.
 * Current implementation: EmailTicketNotificationAdapter; another provider replaces it without touching callers.
 */
public interface TicketNotificationPort {

    /**
     * Tell the customer their ticket was resolved.
     *
     * @param contactEmail    the ticket's contact e-mail
     * @param ticketReference the ticket's public reference
     */
    void notifyResolved(String contactEmail, String ticketReference);
}
