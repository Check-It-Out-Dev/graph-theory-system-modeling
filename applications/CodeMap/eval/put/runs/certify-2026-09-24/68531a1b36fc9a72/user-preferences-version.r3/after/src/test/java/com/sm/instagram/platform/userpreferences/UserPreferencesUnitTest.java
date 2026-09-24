package com.sm.instagram.platform.userpreferences;

import com.sm.instagram.platform.common.util.mappers.EnumTranslationService;
import com.sm.instagram.platform.common.util.mappers.UpdaterIdConverter;
import jakarta.persistence.Version;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.modelmapper.ModelMapper;

import java.lang.reflect.Field;

import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(MockitoExtension.class)
@DisplayName("UserPreferences Version Field Unit Tests")
class UserPreferencesUnitTest {

    @Mock
    private EnumTranslationService enumTranslationService;

    private UserPreferencesMapping userPreferencesMapping;

    @BeforeEach
    void setUp() {
        userPreferencesMapping = new UserPreferencesMapping(enumTranslationService, new UpdaterIdConverter());
    }

    @Nested
    @DisplayName("Version field declaration")
    class VersionFieldDeclarationTests {

        @Test
        @DisplayName("should carry @Version annotation on a Long field for optimistic locking")
        void shouldCarryVersionAnnotationOnLongField() throws NoSuchFieldException {
            Field versionField = UserPreferences.class.getDeclaredField("version");

            assertThat(versionField.isAnnotationPresent(Version.class)).isTrue();
            assertThat(versionField.getType()).isEqualTo(Long.class);
        }
    }

    @Nested
    @DisplayName("Version preservation through updates")
    class VersionPreservationTests {

        @Test
        @DisplayName("should preserve the loaded version when mapping DtoIn onto an existing entity")
        void shouldPreserveVersionWhenMappingDtoInOntoExistingEntity() {
            ModelMapper modelMapper = new ModelMapper();
            userPreferencesMapping.configureMapping(modelMapper);

            UserPreferences existing = new UserPreferences();
            existing.setId(42L);
            existing.setVersion(7L);
            existing.setLanguage("en");

            UserPreferencesDtoIn dtoIn = new UserPreferencesDtoIn();
            dtoIn.setLanguage("pl");
            dtoIn.setTimezone("Europe/Warsaw");

            modelMapper.map(dtoIn, existing);

            assertThat(existing.getVersion()).isEqualTo(7L);
            assertThat(existing.getLanguage()).isEqualTo("pl");
            assertThat(existing.getTimezone()).isEqualTo("Europe/Warsaw");
        }

        @Test
        @DisplayName("should leave version null for a brand-new entity created from DtoIn")
        void shouldLeaveVersionNullForNewEntityFromDtoIn() {
            ModelMapper modelMapper = new ModelMapper();
            userPreferencesMapping.configureMapping(modelMapper);

            UserPreferencesDtoIn dtoIn = new UserPreferencesDtoIn();
            dtoIn.setLanguage("pl");

            UserPreferences created = modelMapper.map(dtoIn, UserPreferences.class);

            assertThat(created.getVersion()).isNull();
        }
    }
}
