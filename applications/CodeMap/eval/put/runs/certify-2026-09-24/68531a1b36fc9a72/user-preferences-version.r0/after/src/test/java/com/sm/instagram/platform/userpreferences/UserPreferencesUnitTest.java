package com.sm.instagram.platform.userpreferences;

import jakarta.persistence.Version;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.modelmapper.ModelMapper;

import java.lang.reflect.Field;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for optimistic locking on UserPreferences.
 * Prevents the web app and mobile app from silently overwriting each other's edits.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("UserPreferences Unit Tests")
class UserPreferencesUnitTest {

    @Test
    @DisplayName("version field carries @Version and is a Long")
    void versionFieldIsAnnotatedForOptimisticLocking() throws NoSuchFieldException {
        Field versionField = UserPreferences.class.getDeclaredField("version");

        assertThat(versionField.isAnnotationPresent(Version.class)).isTrue();
        assertThat(versionField.getType()).isEqualTo(Long.class);
    }

    @Test
    @DisplayName("mapping a DtoIn onto an existing entity preserves its loaded version")
    void mappingDtoInPreservesVersion() {
        UserPreferences preferences = new UserPreferences();
        preferences.setVersion(7L);
        preferences.setLanguage("en");

        UserPreferencesDtoIn dtoIn = new UserPreferencesDtoIn();
        dtoIn.setLanguage("pl");
        dtoIn.setTimezone("Europe/Warsaw");

        new ModelMapper().map(dtoIn, preferences);

        assertThat(preferences.getVersion()).isEqualTo(7L);
        assertThat(preferences.getLanguage()).isEqualTo("pl");
    }
}
