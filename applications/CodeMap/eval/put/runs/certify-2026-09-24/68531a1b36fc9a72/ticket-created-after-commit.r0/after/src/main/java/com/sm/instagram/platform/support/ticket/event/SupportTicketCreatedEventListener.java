package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.common.EmailService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

/**
 * Listens for SupportTicketCreatedEvent and sends the confirmation e-mail.
 * AFTER_COMMIT ensures the ticket is persisted before the send attempt, so a
 * slow mail server can no longer hold the ticket-creation transaction open,
 * and a rollback can no longer leave a customer holding a reference to a
 * ticket that does not exist.
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
                    event.getStatusToken()
            );
            log.info("Sent ticket creation confirmation: ticketId={}, ticketRef={}",
                    event.getTicketId(), event.getTicketReference());
        } catch (Exception e) {
            log.warn("Failed to send ticket creation confirmation: ticketId={}, ticketRef={}, error={}",
                    event.getTicketId(), event.getTicketReference(), e.getMessage());
        }
    }
}
