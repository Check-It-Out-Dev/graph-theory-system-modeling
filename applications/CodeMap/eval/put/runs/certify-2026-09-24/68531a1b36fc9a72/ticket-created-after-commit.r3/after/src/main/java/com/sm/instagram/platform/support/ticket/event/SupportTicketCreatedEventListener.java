package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.support.common.EmailService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

/**
 * Listens for SupportTicketCreatedEvent and sends the confirmation email.
 * AFTER_COMMIT ensures the ticket is persisted before the confirmation (and its
 * magic-link status token) goes out — a slow mail server can no longer hold the
 * ticket-creation transaction open.
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
        } catch (Exception e) {
            log.error("Failed to send ticket creation confirmation: ticketId={}, error={}",
                    event.getTicketId(), e.getMessage(), e);
        }
    }
}
