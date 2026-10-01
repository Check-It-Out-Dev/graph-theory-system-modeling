package com.sm.instagram.platform.support.ticket.dtos;

import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.Map;

/**
 * DTO for outgoing support ticket statistics.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class TicketStatsDtoOut {

    /**
     * Number of tickets created in the period, per status. Every status is
     * present, with 0 when there are no tickets in that status.
     */
    private Map<TicketStatus, Long> countsByStatus;

    /**
     * Total number of tickets created in the period, across all statuses.
     */
    private long total;
}
