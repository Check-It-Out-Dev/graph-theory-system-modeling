package com.sm.instagram.platform.support.ticket.port;

/**
 * Port for notifying a customer about their support ticket.
 * Abstracted to allow swapping the delivery channel (e-mail today, another
 * provider later) without changing business logic.
 */
public interface TicketNotificationPort {

    /**
     * Notify the customer that their ticket has been resolved.
     *
     * @param contactEmail    the customer's contact e-mail
     * @param ticketReference the human-readable ticket reference
     */
    void notifyResolved(String contactEmail, String ticketReference);
}
