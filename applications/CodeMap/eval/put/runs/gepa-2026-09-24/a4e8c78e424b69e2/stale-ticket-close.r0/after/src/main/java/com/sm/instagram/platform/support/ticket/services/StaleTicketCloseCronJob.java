package com.sm.instagram.platform.support.ticket.services;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.javacrumbs.shedlock.spring.annotation.SchedulerLock;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * Nightly cron job that closes support tickets that have stayed RESOLVED
 * for longer than the configured number of days, because nobody reopens
 * them and they should not stay RESOLVED forever.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class StaleTicketCloseCronJob {

    private final SupportTicketService supportTicketService;

    @Value("${support.ticket.stale-close.enabled:true}")
    private boolean enabled;

    @Value("${support.ticket.stale-close.days:14}")
    private int olderThanDays;

    @Scheduled(cron = "${support.ticket.stale-close.cron:0 30 3 * * *}")
    @SchedulerLock(
            name = "support:staleTicketClose",
            lockAtMostFor = "30m",
            lockAtLeastFor = "5m"
    )
    public void closeStaleTickets() {
        if (!enabled) {
            log.debug("Stale ticket close cron job disabled via configuration");
            return;
        }

        log.info("Starting stale resolved ticket close, olderThanDays={}", olderThanDays);

        try {
            int closed = supportTicketService.closeStaleResolvedTickets(olderThanDays);
            log.info("Stale resolved ticket close completed: closed={}", closed);
        } catch (Exception e) {
            log.error("Stale resolved ticket close failed: {}", e.getMessage(), e);
        }
    }
}
