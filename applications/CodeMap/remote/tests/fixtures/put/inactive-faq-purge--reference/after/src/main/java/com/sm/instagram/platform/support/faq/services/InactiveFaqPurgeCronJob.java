package com.sm.instagram.platform.support.faq.services;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.javacrumbs.shedlock.spring.annotation.SchedulerLock;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * Weekly job that deletes FAQs which have been inactive for longer than
 * {@code support.faq.purge.days} days (default 180).
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class InactiveFaqPurgeCronJob {

    private final FaqService faqService;

    @Value("${support.faq.purge.enabled:true}")
    private boolean enabled;

    @Value("${support.faq.purge.days:180}")
    private int olderThanDays;

    @Scheduled(cron = "${support.faq.purge.cron:0 0 4 * * SUN}")
    @SchedulerLock(name = "support:inactiveFaqPurge", lockAtMostFor = "30m", lockAtLeastFor = "1m")
    public void purgeInactiveFaqs() {
        if (!enabled) {
            log.debug("Inactive FAQ purge job disabled via configuration");
            return;
        }
        try {
            int deleted = faqService.purgeInactiveOlderThan(olderThanDays);
            log.info("Inactive FAQ purge job finished: deleted={}", deleted);
        } catch (Exception e) {
            log.error("Inactive FAQ purge job failed: {}", e.getMessage(), e);
        }
    }
}
