package com.sm.instagram.platform.support.ticket.dtos;

import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.Map;

/**
 * DTO for admin ticket-creation statistics over a date range.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class TicketStatsDtoOut {

    /**
     * Number of tickets created in the period, keyed by status. Every
     * {@link TicketStatus} is present, with 0 when none were created.
     */
    private Map<TicketStatus, Long> countsByStatus;

    /**
     * Total number of tickets created in the period, across all statuses.
     */
    private long total;
}
