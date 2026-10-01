package com.sm.instagram.platform.support.ticket.port;

/**
 * Port for notifying a customer that their support ticket was resolved.
 * Primary implementation: {@link com.sm.instagram.platform.support.ticket.adapter.EmailTicketNotificationAdapter} (e-mail).
 * Swappable by implementing this interface in a new adapter without changing SupportTicketService.
 */
public interface TicketNotificationPort {

    /**
     * Notifies the customer that their ticket has been resolved.
     *
     * @param contactEmail    customer's contact e-mail address
     * @param ticketReference human-readable reference of the resolved ticket
     */
    void notifyResolved(String contactEmail, String ticketReference);
}
