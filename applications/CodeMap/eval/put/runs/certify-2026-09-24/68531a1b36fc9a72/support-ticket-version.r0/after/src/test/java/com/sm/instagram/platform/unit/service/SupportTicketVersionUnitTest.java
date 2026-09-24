package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import jakarta.persistence.Version;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for optimistic locking on SupportTicket.
 * Guards against the lost-update race between an admin status change and a customer reply.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SupportTicket Version Field Unit Tests")
class SupportTicketVersionUnitTest {

    @Test
    @DisplayName("version field is annotated with @Version and typed as Long")
    void versionFieldCarriesVersionAnnotationAndIsLong() throws NoSuchFieldException {
        Field versionField = SupportTicket.class.getDeclaredField("version");

        assertThat(versionField.isAnnotationPresent(Version.class)).isTrue();
        assertThat(versionField.getType()).isEqualTo(Long.class);
    }
}
