package com.sm.instagram.platform.support.ticket.port;

/**
 * Port for notifying a customer that their support ticket was resolved.
 * Primary implementation: e-mail. Abstracted to allow swapping the
 * notification provider without changing business logic.
 */
public interface TicketNotificationPort {

    /**
     * Notify the customer that their ticket has been resolved.
     *
     * @param contactEmail    the customer's contact e-mail
     * @param ticketReference the ticket's human-readable reference
     */
    void notifyResolved(String contactEmail, String ticketReference);
}
