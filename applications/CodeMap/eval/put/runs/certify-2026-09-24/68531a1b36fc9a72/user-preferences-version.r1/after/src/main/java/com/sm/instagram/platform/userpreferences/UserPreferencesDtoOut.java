package com.sm.instagram.platform.userpreferences;

import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
@JsonInclude(JsonInclude.Include.NON_NULL)
public class UserPreferencesDtoOut {
    private Long id;
    private Long userId;
    private Boolean notificationEmailEnabled;
    private Boolean notificationPushEnabled;
    private Boolean notificationSmsEnabled;
    private Boolean notificationPartnershipEnabled;
    private Boolean notificationSupportEnabled;
    private Boolean notificationSystemEnabled;
    private Boolean notificationEmailPartnershipEnabled;
    private Boolean notificationEmailSupportEnabled;
    private Boolean darkModeEnabled;
    @Schema(allowableValues = {"pl", "en"}, description = "UI language code")
    private String language;
    private String timezone;
    @Schema(allowableValues = {"IMMEDIATE", "HOURLY_DIGEST", "DAILY_DIGEST", "WEEKLY_DIGEST"}, description = "EmailFrequency enum name")
    private String communicationFrequency;
    private Boolean gdprMarketingConsent;
    private Boolean sharePhoneForPayments;
    private Boolean twoFactorAuthenticationEnabled;
    private LocalDateTime createdTime;
    private LocalDateTime lastUpdateTime;
    private String updaterId;
    private Long version;
}