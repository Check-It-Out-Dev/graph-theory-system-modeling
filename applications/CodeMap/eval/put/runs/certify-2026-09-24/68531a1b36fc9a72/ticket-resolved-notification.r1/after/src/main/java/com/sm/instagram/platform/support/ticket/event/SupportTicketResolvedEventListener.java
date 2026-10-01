package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

/**
 * Listens for SupportTicketResolvedEvent and notifies the customer.
 * AFTER_COMMIT ensures the RESOLVED status is persisted before we notify.
 * Failures are caught and logged — the status change itself must not fail.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class SupportTicketResolvedEventListener {

    private final TicketNotificationPort ticketNotificationPort;

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onTicketResolved(SupportTicketResolvedEvent event) {
        try {
            ticketNotificationPort.notifyResolved(event.getContactEmail(), event.getTicketReference());
            log.info("Ticket resolved notification sent: ticketId={}, ticketRef={}",
                    event.getTicketId(), event.getTicketReference());
        } catch (Exception e) {
            log.error("Failed to send ticket resolved notification: ticketId={}, ticketRef={}, error={}",
                    event.getTicketId(), event.getTicketReference(), e.getMessage(), e);
        }
    }
}
