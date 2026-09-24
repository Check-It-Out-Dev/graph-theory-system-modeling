package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import jakarta.persistence.Column;
import jakarta.persistence.Version;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests protecting SupportTicket's optimistic locking field.
 * Fails if the @Version annotation or its Long type is removed from the entity.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicket Version Unit Tests")
class SupportTicketVersionUnitTest {

    @Test
    @DisplayName("version field is annotated with @Version and typed as Long")
    void versionFieldCarriesVersionAnnotationAndLongType() throws NoSuchFieldException {
        Field versionField = SupportTicket.class.getDeclaredField("version");

        assertThat(versionField.isAnnotationPresent(Version.class)).isTrue();
        assertThat(versionField.getType()).isEqualTo(Long.class);
    }

    @Test
    @DisplayName("version field maps to the version column")
    void versionFieldMapsToVersionColumn() throws NoSuchFieldException {
        Field versionField = SupportTicket.class.getDeclaredField("version");

        Column column = versionField.getAnnotation(Column.class);
        assertThat(column).isNotNull();
        assertThat(column.name()).isEqualTo("version");
    }

    @Test
    @DisplayName("version is null until the entity is persisted")
    void versionIsNullBeforePersistence() {
        SupportTicket ticket = new SupportTicket();

        assertThat(ticket.getVersion()).isNull();
    }

    @Test
    @DisplayName("version can be set to reflect a persisted row")
    void versionReflectsPersistedRow() {
        SupportTicket ticket = new SupportTicket();

        ticket.setVersion(7L);

        assertThat(ticket.getVersion()).isEqualTo(7L);
    }
}
