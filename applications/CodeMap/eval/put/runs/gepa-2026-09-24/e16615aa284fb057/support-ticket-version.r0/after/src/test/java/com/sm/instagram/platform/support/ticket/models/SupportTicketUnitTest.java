package com.sm.instagram.platform.support.ticket.models;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * Unit test for {@link SupportTicket} — the optimistic-locking version field
 * that protects a ticket from lost updates when an admin and the customer
 * change it at the same time.
 */
class SupportTicketUnitTest {

    @Test
    void newTicketHasNullVersionUntilPersisted() {
        SupportTicket ticket = new SupportTicket();

        assertNull(ticket.getVersion());
    }

    @Test
    void versionCanBeSetAndReadBack() {
        SupportTicket ticket = new SupportTicket();

        ticket.setVersion(7L);

        assertEquals(7L, ticket.getVersion());
    }
}
