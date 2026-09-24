package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.user.User;
import com.sm.instagram.platform.userpreferences.UserPreferences;
import com.sm.instagram.platform.userpreferences.UserPreferencesDtoIn;
import com.sm.instagram.platform.userpreferences.UserPreferencesDtoOut;
import jakarta.persistence.Version;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.modelmapper.ModelMapper;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;

import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(MockitoExtension.class)
@DisplayName("UserPreferences optimistic locking Unit Tests")
class UserPreferencesVersionUnitTest {

    @Test
    @DisplayName("version field carries @Version and is a Long")
    void versionFieldCarriesVersionAnnotationAndIsLong() throws NoSuchFieldException {
        // Given
        Field versionField = UserPreferences.class.getDeclaredField("version");

        // Then
        assertThat(versionField.getType()).isEqualTo(Long.class);
        assertThat(versionField.getAnnotation(Version.class)).isNotNull();
    }

    @Test
    @DisplayName("mapping a DtoIn onto an existing entity preserves its version")
    void mappingDtoInOntoExistingEntityPreservesVersion() {
        // Given
        User user = new User();
        user.setId(1L);

        UserPreferences preferences = new UserPreferences();
        preferences.setId(1L);
        preferences.setUser(user);
        preferences.setVersion(7L);
        preferences.setLanguage("en");
        preferences.setDarkModeEnabled(false);

        UserPreferencesDtoIn dtoIn = new UserPreferencesDtoIn();
        dtoIn.setLanguage("pl");
        dtoIn.setDarkModeEnabled(true);

        // When - exactly what UserPreferencesService#updateCurrentUserPreferences does
        new ModelMapper().map(dtoIn, preferences);

        // Then - the fields the client sent are applied, but the version is untouched by the mapper
        assertThat(preferences.getLanguage()).isEqualTo("pl");
        assertThat(preferences.getDarkModeEnabled()).isTrue();
        assertThat(preferences.getVersion()).isEqualTo(7L);
    }

    @Test
    @DisplayName("mapping an entity to its DtoOut carries the version to the client")
    void mappingEntityToDtoOutCarriesVersion() {
        // Given
        User user = new User();
        user.setId(2L);

        UserPreferences preferences = new UserPreferences();
        preferences.setId(2L);
        preferences.setUser(user);
        preferences.setVersion(42L);

        // When
        UserPreferencesDtoOut dtoOut = new ModelMapper().map(preferences, UserPreferencesDtoOut.class);

        // Then
        assertThat(dtoOut.getVersion()).isEqualTo(42L);
    }
}
