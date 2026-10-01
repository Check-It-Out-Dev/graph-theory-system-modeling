package com.sm.instagram.platform.support.ticket.port;

/**
 * Port for notifying a customer about their support ticket.
 * Abstracted to allow swapping the notification channel/provider without
 * changing business logic. Primary implementation: e-mail.
 */
public interface TicketNotificationPort {

    /**
     * Notify the customer that their support ticket has been resolved.
     *
     * @param contactEmail    the customer's contact e-mail
     * @param ticketReference the ticket reference code
     */
    void notifyResolved(String contactEmail, String ticketReference);
}
