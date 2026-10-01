package com.sm.instagram.platform.support.ticket.dtos;

import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.Map;

/**
 * DTO for ticket creation statistics over a period, grouped by status.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class TicketStatsDtoOut {

    /**
     * Number of tickets created in the period, per status. Every status is present, 0 when none.
     */
    private Map<TicketStatus, Long> countsByStatus;

    /**
     * Total number of tickets created in the period, across all statuses.
     */
    private long total;
}
