package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.ticket.port.TicketNotificationPort;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

/**
 * Tells the customer their ticket is resolved, after the status change has committed.
 * A failure is logged and not rethrown: the status change is already saved.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class SupportTicketResolvedEventListener {

    private final TicketNotificationPort port;

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onTicketResolved(SupportTicketResolvedEvent event) {
        try {
            port.notifyResolved(event.getContactEmail(), event.getTicketReference());
        } catch (Exception e) {
            log.warn("Resolved-ticket notice failed: ticketId={}, error={}", event.getTicketId(), e.getMessage());
        }
    }
}
