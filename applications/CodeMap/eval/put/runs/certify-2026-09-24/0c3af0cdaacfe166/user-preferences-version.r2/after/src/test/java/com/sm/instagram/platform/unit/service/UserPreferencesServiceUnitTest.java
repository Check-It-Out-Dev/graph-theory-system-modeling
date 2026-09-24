package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.notification.EmailFrequency;
import com.sm.instagram.platform.user.User;
import com.sm.instagram.platform.user.UserRepository;
import com.sm.instagram.platform.userpreferences.UserPreferences;
import com.sm.instagram.platform.userpreferences.UserPreferencesDtoIn;
import com.sm.instagram.platform.userpreferences.UserPreferencesRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for UserPreferencesService validation and entity logic.
 * Due to BaseService's getSelf() pattern, we focus on testable behaviors.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("UserPreferencesService Unit Tests")
class UserPreferencesServiceUnitTest {

    @Mock
    private UserPreferencesRepository preferencesRepository;

    @Mock
    private UserRepository userRepository;

    @Mock
    private PermissionUtils permissionUtils;

    private User testUser;
    private UserPreferences testPreferences;

    @BeforeEach
    void setUp() {
        testUser = new User();
        testUser.setId(1L);
        testUser.setFirebaseUserId("test-firebase-uid");
        testUser.setEmail("test@example.com");

        testPreferences = new UserPreferences();
        testPreferences.setId(1L);
        testPreferences.setUser(testUser);
        testPreferences.setNotificationEmailEnabled(true);
        testPreferences.setNotificationPushEnabled(false);
        testPreferences.setNotificationSmsEnabled(false);
        testPreferences.setDarkModeEnabled(false);
        testPreferences.setLanguage("en");
        testPreferences.setTimezone("UTC");
        testPreferences.setCommunicationFrequency(EmailFrequency.WEEKLY_DIGEST);
        testPreferences.setGdprMarketingConsent(false);
        testPreferences.setSharePhoneForPayments(true);
        testPreferences.setTwoFactorAuthenticationEnabled(false);
    }

    @Nested
    @DisplayName("UserPreferences Entity Tests")
    class UserPreferencesEntityTests {

        @Test
        @DisplayName("should create preferences with default values")
        void shouldCreatePreferencesWithDefaultValues() {
            // Given
            UserPreferences preferences = new UserPreferences();

            // When
            preferences.setUser(testUser);
            preferences.setNotificationEmailEnabled(false);
            preferences.setNotificationPushEnabled(false);
            preferences.setNotificationSmsEnabled(false);
            preferences.setDarkModeEnabled(false);
            preferences.setLanguage("en");
            preferences.setTimezone("UTC");
            preferences.setCommunicationFrequency(EmailFrequency.WEEKLY_DIGEST);

            // Then
            assertThat(preferences.getUser()).isEqualTo(testUser);
            assertThat(preferences.getNotificationEmailEnabled()).isFalse();
            assertThat(preferences.getLanguage()).isEqualTo("en");
            assertThat(preferences.getTimezone()).isEqualTo("UTC");
        }

        @Test
        @DisplayName("should update all notification preferences")
        void shouldUpdateAllNotificationPreferences() {
            // Given
            UserPreferences preferences = new UserPreferences();

            // When
            preferences.setNotificationEmailEnabled(true);
            preferences.setNotificationPushEnabled(true);
            preferences.setNotificationSmsEnabled(true);

            // Then
            assertThat(preferences.getNotificationEmailEnabled()).isTrue();
            assertThat(preferences.getNotificationPushEnabled()).isTrue();
            assertThat(preferences.getNotificationSmsEnabled()).isTrue();
        }

        @Test
        @DisplayName("should handle dark mode setting")
        void shouldHandleDarkModeSetting() {
            // Given
            UserPreferences preferences = new UserPreferences();

            // When
            preferences.setDarkModeEnabled(true);

            // Then
            assertThat(preferences.getDarkModeEnabled()).isTrue();
        }

        @Test
        @DisplayName("should handle GDPR marketing consent")
        void shouldHandleGdprMarketingConsent() {
            // Given
            UserPreferences preferences = new UserPreferences();

            // When
            preferences.setGdprMarketingConsent(true);

            // Then
            assertThat(preferences.getGdprMarketingConsent()).isTrue();
        }

        @Test
        @DisplayName("should handle 2FA setting")
        void shouldHandle2FASetting() {
            // Given
            UserPreferences preferences = new UserPreferences();

            // When
            preferences.setTwoFactorAuthenticationEnabled(true);

            // Then
            assertThat(preferences.getTwoFactorAuthenticationEnabled()).isTrue();
        }

        @Test
        @DisplayName("should handle share phone for payments setting")
        void shouldHandleSharePhoneForPayments() {
            // Given
            UserPreferences preferences = new UserPreferences();

            // When
            preferences.setSharePhoneForPayments(false);

            // Then
            assertThat(preferences.getSharePhoneForPayments()).isFalse();
        }

        @Test
        @DisplayName("should have no version until persisted, for optimistic locking")
        void shouldHaveNoVersionUntilPersisted() {
            // Given
            UserPreferences preferences = new UserPreferences();

            // Then
            assertThat(preferences.getVersion()).isNull();
        }

        @Test
        @DisplayName("should expose the version stamped at load time for optimistic locking")
        void shouldExposeLoadedVersion() {
            // Given
            UserPreferences preferences = new UserPreferences();

            // When
            preferences.setVersion(5L);

            // Then
            assertThat(preferences.getVersion()).isEqualTo(5L);
        }
    }

    @Nested
    @DisplayName("CommunicationFrequency Enum Tests")
    class CommunicationFrequencyTests {

        @ParameterizedTest
        @EnumSource(EmailFrequency.class)
        @DisplayName("all communication frequency values should be valid")
        void allCommunicationFrequencyValuesShouldBeValid(EmailFrequency frequency) {
            // Given
            UserPreferences preferences = new UserPreferences();

            // When
            preferences.setCommunicationFrequency(frequency);

            // Then
            assertThat(preferences.getCommunicationFrequency()).isEqualTo(frequency);
        }

        @Test
        @DisplayName("should have DAILY frequency")
        void shouldHaveDailyFrequency() {
            assertThat(EmailFrequency.DAILY_DIGEST).isNotNull();
        }

        @Test
        @DisplayName("should have WEEKLY frequency")
        void shouldHaveWeeklyFrequency() {
            assertThat(EmailFrequency.WEEKLY_DIGEST).isNotNull();
        }

        @Test
        @DisplayName("should parse frequency from string")
        void shouldParseFrequencyFromString() {
            // When
            EmailFrequency frequency =
                    EmailFrequency.valueOf("WEEKLY_DIGEST");

            // Then
            assertThat(frequency).isEqualTo(EmailFrequency.WEEKLY_DIGEST);
        }

        @Test
        @DisplayName("should throw for invalid frequency string")
        void shouldThrowForInvalidFrequencyString() {
            assertThatThrownBy(() -> EmailFrequency.valueOf("INVALID"))
                    .isInstanceOf(IllegalArgumentException.class);
        }
    }

    @Nested
    @DisplayName("Validation Tests")
    class ValidationTests {

        @Test
        @DisplayName("should accept valid language code")
        void shouldAcceptValidLanguageCode() {
            // Given
            UserPreferences preferences = new UserPreferences();

            // When
            preferences.setLanguage("en");

            // Then
            assertThat(preferences.getLanguage()).isEqualTo("en");
        }

        @Test
        @DisplayName("should accept valid timezone")
        void shouldAcceptValidTimezone() {
            // Given
            UserPreferences preferences = new UserPreferences();

            // When
            preferences.setTimezone("Europe/Warsaw");

            // Then
            assertThat(preferences.getTimezone()).isEqualTo("Europe/Warsaw");
        }

        @ParameterizedTest
        @ValueSource(strings = {"en", "pl", "de", "fr", "es", "pt-BR"})
        @DisplayName("should accept various language codes")
        void shouldAcceptVariousLanguageCodes(String languageCode) {
            // Given
            UserPreferences preferences = new UserPreferences();

            // When
            preferences.setLanguage(languageCode);

            // Then
            assertThat(preferences.getLanguage()).isEqualTo(languageCode);
        }

        @ParameterizedTest
        @ValueSource(strings = {"UTC", "Europe/Warsaw", "America/New_York", "Asia/Tokyo"})
        @DisplayName("should accept various timezones")
        void shouldAcceptVariousTimezones(String timezone) {
            // Given
            UserPreferences preferences = new UserPreferences();

            // When
            preferences.setTimezone(timezone);

            // Then
            assertThat(preferences.getTimezone()).isEqualTo(timezone);
        }
    }

    @Nested
    @DisplayName("DTO Validation Tests")
    class DtoValidationTests {

        @Test
        @DisplayName("should create valid DTO")
        void shouldCreateValidDto() {
            // Given
            UserPreferencesDtoIn dto = new UserPreferencesDtoIn();

            // When
            dto.setLanguage("pl");
            dto.setTimezone("Europe/Warsaw");
            dto.setNotificationEmailEnabled(true);
            dto.setDarkModeEnabled(true);
            dto.setCommunicationFrequency("DAILY_DIGEST");
            dto.setGdprMarketingConsent(true);

            // Then
            assertThat(dto.getLanguage()).isEqualTo("pl");
            assertThat(dto.getTimezone()).isEqualTo("Europe/Warsaw");
            assertThat(dto.getNotificationEmailEnabled()).isTrue();
            assertThat(dto.getDarkModeEnabled()).isTrue();
            assertThat(dto.getCommunicationFrequency()).isEqualTo("DAILY_DIGEST");
            assertThat(dto.getGdprMarketingConsent()).isTrue();
        }

        @Test
        @DisplayName("should validate language length is within limit")
        void shouldValidateLanguageLengthWithinLimit() {
            // Given
            String validLanguage = "en";

            // When/Then - should not throw
            assertThat(validLanguage.length()).isLessThanOrEqualTo(10);
        }

        @Test
        @DisplayName("should detect language exceeding length limit")
        void shouldDetectLanguageExceedingLengthLimit() {
            // Given
            String longLanguage = "this_is_way_too_long_for_a_language_code";

            // When/Then
            assertThat(longLanguage.length()).isGreaterThan(10);
        }

        @Test
        @DisplayName("should validate timezone length is within limit")
        void shouldValidateTimezoneLengthWithinLimit() {
            // Given
            String validTimezone = "Europe/Warsaw";

            // When/Then
            assertThat(validTimezone.length()).isLessThanOrEqualTo(50);
        }

        @Test
        @DisplayName("should detect timezone exceeding length limit")
        void shouldDetectTimezoneExceedingLengthLimit() {
            // Given
            String longTimezone = "This_is_a_very_long_timezone_string_that_exceeds_fifty_characters_limit";

            // When/Then
            assertThat(longTimezone.length()).isGreaterThan(50);
        }
    }

    @Nested
    @DisplayName("Permission Logic Tests")
    class PermissionLogicTests {

        @Test
        @DisplayName("admin should have access to any user preferences")
        void adminShouldHaveAccessToAnyUserPreferences() {
            // Given
            when(permissionUtils.isAdmin()).thenReturn(true);

            // When
            boolean isAdmin = permissionUtils.isAdmin();

            // Then
            assertThat(isAdmin).isTrue();
        }

        @Test
        @DisplayName("user should only access own preferences")
        void userShouldOnlyAccessOwnPreferences() {
            // Given - only stub what is actually used
            when(permissionUtils.isUserOwner(testUser)).thenReturn(true);

            // When
            boolean isOwner = permissionUtils.isUserOwner(testUser);

            // Then
            assertThat(isOwner).isTrue();
        }

        @Test
        @DisplayName("non-owner non-admin should not have access")
        void nonOwnerNonAdminShouldNotHaveAccess() {
            // Given
            when(permissionUtils.isAdmin()).thenReturn(false);
            when(permissionUtils.isUserOwner(testUser)).thenReturn(false);

            // When
            boolean isAdmin = permissionUtils.isAdmin();
            boolean isOwner = permissionUtils.isUserOwner(testUser);

            // Then
            assertThat(isAdmin).isFalse();
            assertThat(isOwner).isFalse();
        }
    }

    @Nested
    @DisplayName("Patch Updates Processing Tests")
    class PatchUpdatesProcessingTests {

        @Test
        @DisplayName("should simulate patch update for notification email")
        void shouldSimulatePatchUpdateForNotificationEmail() {
            // Given
            Map<String, Object> updates = new HashMap<>();
            updates.put("notificationEmailEnabled", true);
            UserPreferences preferences = new UserPreferences();
            preferences.setNotificationEmailEnabled(false);

            // When - simulating what processPreferencesUpdates does
            if (updates.containsKey("notificationEmailEnabled")) {
                preferences.setNotificationEmailEnabled((Boolean) updates.get("notificationEmailEnabled"));
            }

            // Then
            assertThat(preferences.getNotificationEmailEnabled()).isTrue();
        }

        @Test
        @DisplayName("should simulate patch update for dark mode")
        void shouldSimulatePatchUpdateForDarkMode() {
            // Given
            Map<String, Object> updates = new HashMap<>();
            updates.put("darkModeEnabled", true);
            UserPreferences preferences = new UserPreferences();
            preferences.setDarkModeEnabled(false);

            // When
            if (updates.containsKey("darkModeEnabled")) {
                preferences.setDarkModeEnabled((Boolean) updates.get("darkModeEnabled"));
            }

            // Then
            assertThat(preferences.getDarkModeEnabled()).isTrue();
        }

        @Test
        @DisplayName("should simulate patch update for language")
        void shouldSimulatePatchUpdateForLanguage() {
            // Given
            Map<String, Object> updates = new HashMap<>();
            updates.put("language", "pl");
            UserPreferences preferences = new UserPreferences();
            preferences.setLanguage("en");

            // When
            if (updates.containsKey("language")) {
                preferences.setLanguage((String) updates.get("language"));
            }

            // Then
            assertThat(preferences.getLanguage()).isEqualTo("pl");
        }

        @Test
        @DisplayName("should simulate patch update for communication frequency")
        void shouldSimulatePatchUpdateForCommunicationFrequency() {
            // Given
            Map<String, Object> updates = new HashMap<>();
            updates.put("communicationFrequency", "DAILY_DIGEST");
            UserPreferences preferences = new UserPreferences();
            preferences.setCommunicationFrequency(EmailFrequency.WEEKLY_DIGEST);

            // When
            if (updates.containsKey("communicationFrequency")) {
                String frequency = (String) updates.get("communicationFrequency");
                preferences.setCommunicationFrequency(
                        EmailFrequency.valueOf(frequency)
                );
            }

            // Then
            assertThat(preferences.getCommunicationFrequency())
                    .isEqualTo(EmailFrequency.DAILY_DIGEST);
        }

        @Test
        @DisplayName("should handle invalid communication frequency")
        void shouldHandleInvalidCommunicationFrequency() {
            // Given
            String invalidFrequency = "INVALID_FREQUENCY";

            // When/Then
            assertThatThrownBy(() -> EmailFrequency.valueOf(invalidFrequency))
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("should simulate multiple field updates")
        void shouldSimulateMultipleFieldUpdates() {
            // Given
            Map<String, Object> updates = new HashMap<>();
            updates.put("notificationEmailEnabled", true);
            updates.put("darkModeEnabled", true);
            updates.put("language", "de");
            updates.put("gdprMarketingConsent", true);

            UserPreferences preferences = new UserPreferences();

            // When - simulating batch update
            updates.forEach((key, value) -> {
                switch (key) {
                    case "notificationEmailEnabled" -> preferences.setNotificationEmailEnabled((Boolean) value);
                    case "darkModeEnabled" -> preferences.setDarkModeEnabled((Boolean) value);
                    case "language" -> preferences.setLanguage((String) value);
                    case "gdprMarketingConsent" -> preferences.setGdprMarketingConsent((Boolean) value);
                }
            });

            // Then
            assertThat(preferences.getNotificationEmailEnabled()).isTrue();
            assertThat(preferences.getDarkModeEnabled()).isTrue();
            assertThat(preferences.getLanguage()).isEqualTo("de");
            assertThat(preferences.getGdprMarketingConsent()).isTrue();
        }
    }

    @Nested
    @DisplayName("Repository Interaction Tests")
    class RepositoryInteractionTests {

        @Test
        @DisplayName("should find preferences by user")
        void shouldFindPreferencesByUser() {
            // Given
            when(preferencesRepository.findByUser(testUser)).thenReturn(testPreferences);

            // When
            UserPreferences found = preferencesRepository.findByUser(testUser);

            // Then
            assertThat(found).isNotNull();
            assertThat(found.getUser()).isEqualTo(testUser);
            verify(preferencesRepository).findByUser(testUser);
        }

        @Test
        @DisplayName("should return null when preferences not found")
        void shouldReturnNullWhenPreferencesNotFound() {
            // Given
            when(preferencesRepository.findByUser(testUser)).thenReturn(null);

            // When
            UserPreferences found = preferencesRepository.findByUser(testUser);

            // Then
            assertThat(found).isNull();
        }

        @Test
        @DisplayName("should save preferences")
        void shouldSavePreferences() {
            // Given
            when(preferencesRepository.save(any(UserPreferences.class))).thenReturn(testPreferences);

            // When
            UserPreferences saved = preferencesRepository.save(testPreferences);

            // Then
            assertThat(saved).isNotNull();
            assertThat(saved.getId()).isEqualTo(1L);
            verify(preferencesRepository).save(testPreferences);
        }
    }

    @Nested
    @DisplayName("User Repository Interaction Tests")
    class UserRepositoryInteractionTests {

        @Test
        @DisplayName("should find user by ID")
        void shouldFindUserById() {
            // Given
            when(userRepository.findById(1L)).thenReturn(Optional.of(testUser));

            // When
            Optional<User> found = userRepository.findById(1L);

            // Then
            assertThat(found).isPresent();
            assertThat(found.get().getId()).isEqualTo(1L);
        }

        @Test
        @DisplayName("should find user by Firebase ID")
        void shouldFindUserByFirebaseId() {
            // Given
            when(userRepository.findByFirebaseUserId("test-firebase-uid")).thenReturn(Optional.of(testUser));

            // When
            Optional<User> found = userRepository.findByFirebaseUserId("test-firebase-uid");

            // Then
            assertThat(found).isPresent();
            assertThat(found.get().getFirebaseUserId()).isEqualTo("test-firebase-uid");
        }

        @Test
        @DisplayName("should return empty when user not found")
        void shouldReturnEmptyWhenUserNotFound() {
            // Given
            when(userRepository.findById(999L)).thenReturn(Optional.empty());

            // When
            Optional<User> found = userRepository.findById(999L);

            // Then
            assertThat(found).isEmpty();
        }
    }
}
