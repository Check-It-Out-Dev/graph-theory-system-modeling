package com.sm.instagram.platform.support.ticket.port;

/**
 * Port for notifying a ticket's contact that their support ticket was resolved.
 * Abstracted to allow swapping the notification provider without changing business logic.
 */
public interface TicketNotificationPort {

    void notifyResolved(String contactEmail, String ticketReference);
}
