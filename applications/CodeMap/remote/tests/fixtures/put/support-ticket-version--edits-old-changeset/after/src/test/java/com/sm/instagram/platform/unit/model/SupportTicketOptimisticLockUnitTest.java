package com.sm.instagram.platform.unit.model;

import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import jakarta.persistence.Column;
import jakarta.persistence.Version;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;

import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(MockitoExtension.class)
class SupportTicketOptimisticLockUnitTest {

    @Test
    void versionFieldIsMappedForOptimisticLocking() throws NoSuchFieldException {
        Field field = SupportTicket.class.getDeclaredField("version");

        assertThat(field.getType()).isEqualTo(Long.class);
        assertThat(field.isAnnotationPresent(Version.class)).isTrue();
        assertThat(field.getAnnotation(Column.class).name()).isEqualTo("version");
    }

    @Test
    void newTicketStartsWithoutVersion() {
        assertThat(new SupportTicket().getVersion()).isNull();
    }
}
