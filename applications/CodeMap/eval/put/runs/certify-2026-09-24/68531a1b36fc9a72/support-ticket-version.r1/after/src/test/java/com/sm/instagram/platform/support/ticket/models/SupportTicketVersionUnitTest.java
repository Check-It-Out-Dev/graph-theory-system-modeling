package com.sm.instagram.platform.support.ticket.models;

import jakarta.persistence.Version;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Confirms SupportTicket carries JPA optimistic locking, so a concurrent
 * admin status change and customer reply cannot silently overwrite each
 * other (lost update).
 */
@ExtendWith(MockitoExtension.class)
class SupportTicketVersionUnitTest {

    @Test
    @DisplayName("version field is annotated with @Version and typed Long")
    void versionFieldCarriesOptimisticLocking() throws NoSuchFieldException {
        Field versionField = SupportTicket.class.getDeclaredField("version");

        assertThat(versionField.isAnnotationPresent(Version.class)).isTrue();
        assertThat(versionField.getType()).isEqualTo(Long.class);
    }

    @Test
    @DisplayName("two concurrently loaded copies of the same ticket start with the same version")
    void concurrentlyLoadedCopiesShareTheSameStartingVersion() {
        SupportTicket adminCopy = new SupportTicket();
        adminCopy.setId(7L);
        adminCopy.setVersion(3L);

        SupportTicket customerCopy = new SupportTicket();
        customerCopy.setId(7L);
        customerCopy.setVersion(3L);

        // Given both copies were loaded before either wrote back, Hibernate's
        // optimistic check for the second writer only fires because the two
        // in-memory copies carry the version they were loaded with.
        assertThat(customerCopy.getVersion()).isEqualTo(adminCopy.getVersion());

        // The admin's write bumps its own in-memory copy; the customer's
        // stale copy still points at the version the row no longer has.
        adminCopy.setVersion(adminCopy.getVersion() + 1);
        assertThat(customerCopy.getVersion()).isNotEqualTo(adminCopy.getVersion());
    }
}
