package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.common.EmailService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

/**
 * Sends the ticket confirmation e-mail after the ticket's transaction has committed.
 * A failure is logged and not rethrown: the ticket is already saved.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class SupportTicketCreatedEventListener {

    private final EmailService emailService;

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onTicketCreated(SupportTicketCreatedEvent event) {
        try {
            emailService.sendTicketCreationConfirmation(
                    event.getContactEmail(),
                    event.getTicketReference(),
                    event.getSubject(),
                    event.getLanguage(),
                    event.getStatusToken());
        } catch (Exception e) {
            log.warn("Ticket confirmation e-mail failed: ticketId={}, error={}", event.getTicketId(), e.getMessage());
        }
    }
}
