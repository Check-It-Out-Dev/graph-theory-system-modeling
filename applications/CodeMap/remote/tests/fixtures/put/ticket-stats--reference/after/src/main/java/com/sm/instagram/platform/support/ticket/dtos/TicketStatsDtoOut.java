package com.sm.instagram.platform.support.ticket.dtos;

import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.Map;

/**
 * Ticket counts per status for a period: every status is present, 0 when none.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class TicketStatsDtoOut {

    private Map<TicketStatus, Long> countsByStatus;
    private long total;
}
