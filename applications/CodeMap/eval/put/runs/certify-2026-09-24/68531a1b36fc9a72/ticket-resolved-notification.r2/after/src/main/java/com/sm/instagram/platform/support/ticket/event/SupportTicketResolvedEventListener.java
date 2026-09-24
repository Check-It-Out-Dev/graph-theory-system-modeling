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
 * Failures are caught and logged — the status change already committed.
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
        } catch (Exception e) {
            log.warn("Failed to notify customer of resolved ticket: ticketId={}, error={}",
                    event.getTicketId(), e.getMessage());
        }
    }
}
