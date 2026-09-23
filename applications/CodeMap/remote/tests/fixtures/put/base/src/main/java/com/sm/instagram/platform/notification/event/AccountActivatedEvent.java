package com.sm.instagram.platform.notification.event;

import com.sm.instagram.platform.user.AccountStatus;
import com.sm.instagram.platform.user.User;
import lombok.Getter;
import org.springframework.context.ApplicationEvent;

import java.time.LocalDateTime;

/**
 * Domain event published when a user's account is activated.
 * <p>
 * Published by: UserService (admin activation), EmailVerificationService (auto-activation),
 *               RegistryLookupService (company data confirmation auto-activation)
 * Handled by: NotificationEventListener
 * <p>
 * IMPORTANT: This event is published BEFORE the transaction commits.
 * The listener uses @TransactionalEventListener(phase = AFTER_COMMIT) to ensure
 * notification creation only happens after the business operation succeeds.
 */
@Getter
public class AccountActivatedEvent extends ApplicationEvent {

    private final User user;
    private final AccountStatus previousStatus;
    private final String activationSource;
    private final LocalDateTime occurredAt;

    /**
     * @param source           the object that published this event
     * @param user             the user whose account was activated
     * @param previousStatus   status before activation
     * @param activationSource how the account was activated (ADMIN, EMAIL_VERIFICATION, COMPANY_DATA_CONFIRMATION)
     */
    public AccountActivatedEvent(Object source, User user, AccountStatus previousStatus, String activationSource) {
        super(source);
        this.user = user;
        this.previousStatus = previousStatus;
        this.activationSource = activationSource;
        this.occurredAt = LocalDateTime.now();
    }

    @Override
    public String toString() {
        return String.format("AccountActivatedEvent{userId=%d, %s→ACTIVE, source=%s}",
                user.getId(), previousStatus, activationSource);
    }
}
