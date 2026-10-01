package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

/**
 * Listens for SupportTicketResolvedEvent and notifies the customer.
 * AFTER_COMMIT ensures the status change is persisted before we notify.
 * Failures are caught and logged — the status update itself has already succeeded.
 */
@Slf4j
@Component
public class SupportTicketResolvedEventListener {

    private final TicketNotificationPort notificationPort;

    public SupportTicketResolvedEventListener(TicketNotificationPort notificationPort) {
        this.notificationPort = notificationPort;
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onTicketResolved(SupportTicketResolvedEvent event) {
        try {
            notificationPort.notifyResolved(event.getContactEmail(), event.getTicketReference());
            log.info("Ticket resolution notification sent: ticketId={}, ticketRef={}",
                    event.getTicketId(), event.getTicketReference());
        } catch (Exception e) {
            log.error("Failed to send ticket resolution notification: ticketId={}, ticketRef={}",
                    event.getTicketId(), event.getTicketReference(), e);
        }
    }
}
