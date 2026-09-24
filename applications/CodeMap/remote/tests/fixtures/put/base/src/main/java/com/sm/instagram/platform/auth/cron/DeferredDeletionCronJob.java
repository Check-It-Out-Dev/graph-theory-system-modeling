package com.sm.instagram.platform.auth.cron;

import com.sm.instagram.platform.auth.entity.DeletionRequestStatus;
import com.sm.instagram.platform.auth.entity.PendingDataDeletionRequest;
import com.sm.instagram.platform.auth.repository.PendingDataDeletionRequestRepository;
import com.sm.instagram.platform.user.User;
import com.sm.instagram.platform.user.UserAccountOrchestrator;
import com.sm.instagram.platform.user.dto.DeletionEligibilityDto;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.javacrumbs.shedlock.spring.annotation.SchedulerLock;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.support.TransactionTemplate;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Locale;

/**
 * Daily cron job that processes deferred data deletion requests.
 * Re-checks whether active collaborations have ended and executes
 * deletion when eligible.
 *
 * Uses ShedLock to prevent concurrent execution in distributed environments.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class DeferredDeletionCronJob {

    private final PendingDataDeletionRequestRepository deletionRequestRepository;
    private final UserAccountOrchestrator userAccountOrchestrator;
    private final TransactionTemplate transactionTemplate;

    @Value("${meta.deletion.enabled:true}")
    private boolean enabled;

    @Scheduled(cron = "${meta.deletion.cron:0 0 3 * * *}")
    @SchedulerLock(
            name = "instagram:deferredDeletionCheck",
            lockAtMostFor = "30m",
            lockAtLeastFor = "5m"
    )
    public void processDeferredDeletions() {
        if (!enabled) {
            log.debug("Deferred deletion cron job disabled via configuration");
            return;
        }

        log.info("GDPR: Starting deferred data deletion check");
        long startTime = System.currentTimeMillis();

        try {
            List<PendingDataDeletionRequest> pendingRequests =
                    deletionRequestRepository.findAllByStatus(DeletionRequestStatus.PENDING);

            log.info("GDPR: Found {} pending data deletion requests", pendingRequests.size());

            int completed = 0;
            int stillBlocked = 0;
            int errors = 0;

            for (PendingDataDeletionRequest request : pendingRequests) {
                try {
                    boolean processed = processOneRequest(request);
                    if (processed) {
                        completed++;
                    } else {
                        stillBlocked++;
                    }
                } catch (Exception e) {
                    errors++;
                    log.error("GDPR: Error processing deferred deletion for request {}: {}",
                            request.getId(), e.getMessage(), e);
                }
            }

            log.info("GDPR: Deferred deletion check: {} completed, {} still blocked, {} errors",
                    completed, stillBlocked, errors);
        } catch (Exception e) {
            log.error("GDPR: Deferred deletion cron job failed: {}", e.getMessage(), e);
        }

        long duration = System.currentTimeMillis() - startTime;
        log.info("GDPR: Deferred data deletion check completed in {}ms", duration);
    }

    private boolean processOneRequest(PendingDataDeletionRequest request) {
        User user = request.getUser();
        DeletionEligibilityDto eligibility =
                userAccountOrchestrator.checkDeletionEligibilityForUser(user, Locale.ENGLISH);

        if (!eligibility.isCanSoftDelete()) {
            log.info("GDPR: Deferred deletion for user {} still blocked by active collaborations", user.getId());
            return false;
        }

        log.info("GDPR: Deferred deletion for user {} — collaborations cleared, executing archival", user.getId());

        transactionTemplate.executeWithoutResult(status -> {
            userAccountOrchestrator.archiveUser(user);

            request.setStatus(DeletionRequestStatus.COMPLETED);
            request.setCompletedAt(LocalDateTime.now());
            request.setBlockers(null);
            deletionRequestRepository.save(request);
        });

        log.info("GDPR: Deferred deletion completed for user {}, confirmation code: {}",
                user.getId(), request.getConfirmationCode());
        return true;
    }
}
