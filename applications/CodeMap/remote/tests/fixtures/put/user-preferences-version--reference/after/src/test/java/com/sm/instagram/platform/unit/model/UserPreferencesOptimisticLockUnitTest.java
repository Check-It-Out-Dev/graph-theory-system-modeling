package com.sm.instagram.platform.unit.model;

import com.sm.instagram.platform.userpreferences.UserPreferences;
import jakarta.persistence.Column;
import jakarta.persistence.Version;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;

import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(MockitoExtension.class)
class UserPreferencesOptimisticLockUnitTest {

    @Test
    void versionFieldIsMappedForOptimisticLocking() throws NoSuchFieldException {
        Field field = UserPreferences.class.getDeclaredField("version");

        assertThat(field.getType()).isEqualTo(Long.class);
        assertThat(field.isAnnotationPresent(Version.class)).isTrue();
        assertThat(field.getAnnotation(Column.class).name()).isEqualTo("version");
    }

    @Test
    void newPreferencesStartWithoutVersion() {
        assertThat(new UserPreferences().getVersion()).isNull();
    }
}
