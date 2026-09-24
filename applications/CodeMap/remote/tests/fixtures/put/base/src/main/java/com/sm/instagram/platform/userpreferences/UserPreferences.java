package com.sm.instagram.platform.userpreferences;

import com.sm.instagram.platform.notification.EmailFrequency;
import com.sm.instagram.platform.user.User;
import jakarta.persistence.*;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

@Getter
@Setter
@Entity
@NoArgsConstructor
@AllArgsConstructor
@Table(name = "user_preferences", schema = "public")
public class UserPreferences {

    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "user_preferences_generator")
    @SequenceGenerator(
            name = "user_preferences_generator",
            sequenceName = "user_preferences_seq",
            schema = "public",
            allocationSize = 50,
            initialValue = 1
    )
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "notification_email_enabled")
    private Boolean notificationEmailEnabled = true;

    @Column(name = "notification_push_enabled")
    private Boolean notificationPushEnabled = false;

    @Column(name = "notification_sms_enabled")
    private Boolean notificationSmsEnabled = false;

    @Column(name = "dark_mode_enabled")
    private Boolean darkModeEnabled = false;

    @Column(name = "language")
    private String language = "en";

    @Column(name = "timezone")
    private String timezone = "UTC";

    @Enumerated(EnumType.STRING)
    @Column(name = "communication_frequency", nullable = false, length = 20)
    private EmailFrequency communicationFrequency = EmailFrequency.WEEKLY_DIGEST;

    @Column(name = "gdpr_marketing_consent")
    private Boolean gdprMarketingConsent = false;

    @Column(name = "share_phone_for_payments")
    private Boolean sharePhoneForPayments = false;

    @Column(name = "two_factor_authentication_enabled")
    private Boolean twoFactorAuthenticationEnabled = false;

    @CreationTimestamp
    @Column(updatable = false)
    private LocalDateTime createdTime;

    @UpdateTimestamp
    @Column(name = "last_update_time")
    private LocalDateTime lastUpdateTime;

    @Size(max = 255)
    @Pattern(regexp = "^[a-zA-Z0-9]+$")
    private String updaterId;

    /**
     * Enable/disable partnership notifications (in-app).
     * Controls: application, content, collaboration notifications.
     */
    @Column(name = "notification_partnership_enabled")
    private Boolean notificationPartnershipEnabled = true;

    /**
     * Enable/disable support notifications (in-app).
     * Controls: ticket updates, admin messages.
     */
    @Column(name = "notification_support_enabled")
    private Boolean notificationSupportEnabled = true;

    /**
     * Enable/disable system notifications (in-app).
     * Cannot actually be disabled - field for consistency.
     */
    @Column(name = "notification_system_enabled")
    private Boolean notificationSystemEnabled = true;

    /**
     * Enable email for partnership category.
     * More granular than global notificationEmailEnabled.
     */
    @Column(name = "notification_email_partnership_enabled")
    private Boolean notificationEmailPartnershipEnabled = true;

    /**
     * Enable email for support category.
     */
    @Column(name = "notification_email_support_enabled")
    private Boolean notificationEmailSupportEnabled = true;
}