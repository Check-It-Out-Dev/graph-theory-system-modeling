package com.sm.instagram.platform.support.ticket.port;

/**
 * Port for notifying a customer that their support ticket was resolved.
 * Today's implementation sends an e-mail; the channel is expected to change.
 */
public interface TicketNotificationPort {

    /**
     * Notify the customer that their ticket has been resolved.
     *
     * @param contactEmail    the customer's contact e-mail
     * @param ticketReference the ticket's reference code
     */
    void notifyResolved(String contactEmail, String ticketReference);
}
