package com.sm.instagram.platform.support.ticket.services;

import org.springframework.beans.factory.annotation.Autowired;
import lombok.extern.slf4j.Slf4j;
import net.javacrumbs.shedlock.spring.annotation.SchedulerLock;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * Nightly job that closes support tickets which have stayed RESOLVED for longer than
 * {@code support.ticket.stale-close.days} days (default 14).
 */
@Slf4j
@Component
public class StaleTicketCloseCronJob {

    @Autowired
    private SupportTicketService supportTicketService;

    @Value("${support.ticket.stale-close.enabled:true}")
    private boolean enabled;

    @Value("${support.ticket.stale-close.days:14}")
    private int olderThanDays;

    @Scheduled(cron = "${support.ticket.stale-close.cron:0 30 3 * * *}")
    @SchedulerLock(name = "support:staleTicketClose", lockAtMostFor = "30m", lockAtLeastFor = "1m")
    public void closeStaleTickets() {
        if (!enabled) {
            log.debug("Stale ticket close job disabled via configuration");
            return;
        }
        try {
            int closed = supportTicketService.closeStaleResolvedTickets(olderThanDays);
            log.info("Stale ticket close job finished: closed={}", closed);
        } catch (Exception e) {
            log.error("Stale ticket close job failed: {}", e.getMessage(), e);
        }
    }
}
