package com.sm.instagram.platform.activecooperations;

import com.fasterxml.jackson.annotation.JsonView;
import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import com.sm.instagram.platform.appliedopportunities.OpportunityStatus;
import com.sm.instagram.platform.appliedopportunities.RateStatus;
import com.sm.instagram.platform.common.ratelimit.RateLimit;
import com.sm.instagram.platform.common.ratelimit.RateLimitKeyType;
import com.sm.instagram.platform.common.ratelimit.RateLimitProfile;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.json.MappingJacksonValue;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * REST endpoints for the "active cooperations" panel. Each endpoint resolves
 * the caller's Firebase UID for GDPR audit logging, clamps requested page
 * sizes to a hard maximum, and delegates to {@link ActiveCooperationService}.
 *
 * <p>The two rating mutations ({@code /{id}/company-rating} and
 * {@code /{id}/influencer-rating}) catch and re-log every exception at ERROR
 * level before re-throwing, so failed updates always leave a forensic trail
 * even after {@code @ControllerAdvice} translates the exception for the
 * client.
 */
@Slf4j
@RestController
@RequestMapping("/activecoop")
@PreAuthorize("hasAuthority('ADMIN') or hasAuthority('COMPANY') or hasAuthority('INFLUENCER')")
@RateLimit(profile = RateLimitProfile.STANDARD, keyType = RateLimitKeyType.USER_ENDPOINT)
@RequiredArgsConstructor
public class ActiveCooperationController {

    private static final int MAX_PAGE_SIZE = 100;
    private static final String SORT_FIELD = "lastUpdateTime";

    private final ActiveCooperationService activeCooperationService;

    @Value("${pagination.default-size:12}")
    private int defaultPageSize;

    /**
     * List completed cooperations awaiting a rating decision.
     *
     * @param filterRateStatus one of {@code DEFAULT}, {@code POSITIVE},
     *                         {@code NEGATIVE}, or {@code ALL} for no filter
     */
    @JsonView(Views.Ratings.class)
    @GetMapping("/rate")
    public ResponseEntity<List<CoopDto>> getInfluencersToRate(
            @RequestParam(required = false, defaultValue = "DEFAULT") String filterRateStatus,
            @RequestParam(required = false, defaultValue = "0") int page,
            @RequestParam(required = false) Integer size) {

        String firebaseUid = currentFirebaseUid();
        log.info("GDPR: Operation=getInfluencersToRate, FirebaseUID={}, FilterStatus={}, Page={}, Purpose=rating_retrieval",
                firebaseUid, filterRateStatus, page);

        Pageable pageable = pageable(page, size, Sort.Direction.ASC);
        Page<CoopDto> pageResult = activeCooperationService
                .getInfluencersToRateWithPermission(filterRateStatus, pageable);

        log.info("GDPR: DataAccessed=influencer_ratings, FirebaseUID={}, RecordsReturned={}, Purpose=display",
                firebaseUid, pageResult.getContent().size());
        return ResponseEntity.ok(pageResult.getContent());
    }

    /**
     * List fresh applications (status {@code APPLIED}) for the company to
     * triage. All filters are optional; passing {@code null} skips them.
     */
    @JsonView(Views.Registration.class)
    @GetMapping("/accept")
    public ResponseEntity<List<CoopDto>> getInfluencersToAccept(
            @RequestParam(required = false) Integer minFollowers,
            @RequestParam(required = false) Integer maxFollowers,
            @RequestParam(required = false) Long minPositiveRates,
            @RequestParam(required = false, defaultValue = "0") int page,
            @RequestParam(required = false, defaultValue = "12") Integer size) {

        String firebaseUid = currentFirebaseUid();
        log.info("GDPR: Operation=getInfluencersToAccept, FirebaseUID={}, MinFollowers={}, MaxFollowers={}, Page={}, Purpose=acceptance_review",
                firebaseUid, minFollowers, maxFollowers, page);

        Pageable pageable = pageable(page, size, Sort.Direction.DESC);
        Page<CoopDto> result = activeCooperationService
                .getInfluencersToAccept(minFollowers, maxFollowers, minPositiveRates, pageable);

        log.info("GDPR: DataAccessed=influencer_registrations, FirebaseUID={}, RecordsReturned={}, Purpose=display",
                firebaseUid, result.getContent().size());
        return ResponseEntity.ok(result.getContent());
    }

    /**
     * List in-flight cooperations. {@code statuses} is optional; when
     * supplied it must be a subset of the in-progress whitelist (validated
     * by the service).
     */
    @GetMapping("/inprogress")
    @ApiResponse(responseCode = "200", description = "Cooperations currently in progress",
            content = @Content(array = @ArraySchema(schema = @Schema(implementation = CoopDto.class))))
    public ResponseEntity<MappingJacksonValue> getOpportunitiesInProgress(
            @RequestParam(required = false, defaultValue = "0") int page,
            @RequestParam(required = false, defaultValue = "12") Integer size,
            @RequestParam(required = false) List<OpportunityStatus> statuses) {

        String firebaseUid = currentFirebaseUid();
        log.info("GDPR: Operation=getOpportunitiesInProgress, FirebaseUID={}, Page={}, Statuses={}, Purpose=status_tracking",
                firebaseUid, page, statuses);

        Pageable pageable = pageable(page, size, Sort.Direction.DESC);
        MappingJacksonValue result = activeCooperationService
                .getCollaborationsInProgress(pageable, statuses);

        log.info("GDPR: DataAccessed=collaborations_in_progress, FirebaseUID={}, Purpose=display",
                firebaseUid);
        return ResponseEntity.ok(result);
    }

    /**
     * Set the company-side rating on an applied opportunity.
     */
    @PutMapping("/{id}/company-rating")
    @ApiResponse(responseCode = "200", description = "The cooperation with the company rating applied",
            content = @Content(schema = @Schema(implementation = CoopDto.class)))
    public ResponseEntity<MappingJacksonValue> updateCompanyRating(
            @PathVariable Long id,
            @RequestParam RateStatus rating) {
        return updateRatingEndpoint(
                "updateCompanyRating",
                id,
                rating,
                activeCooperationService::updateCompanyRating,
                Views.InProgress_CompanyView.class);
    }

    /**
     * Set the influencer-side rating on an applied opportunity.
     */
    @PutMapping("/{id}/influencer-rating")
    @ApiResponse(responseCode = "200", description = "The cooperation with the influencer rating applied",
            content = @Content(schema = @Schema(implementation = CoopDto.class)))
    public ResponseEntity<MappingJacksonValue> updateInfluencerRating(
            @PathVariable Long id,
            @RequestParam RateStatus rating) {
        return updateRatingEndpoint(
                "updateInfluencerRating",
                id,
                rating,
                activeCooperationService::updateInfluencerRating,
                Views.InProgress_InfluencerView.class);
    }

    // ----- private helpers -----

    private ResponseEntity<MappingJacksonValue> updateRatingEndpoint(
            String operation,
            Long opportunityId,
            RateStatus rating,
            java.util.function.BiFunction<Long, RateStatus, CoopDto> serviceCall,
            Class<?> viewClass) {

        String firebaseUid = currentFirebaseUid();
        log.warn("GDPR: UPDATE Operation={}, FirebaseUID={}, OpportunityID={}, Rating={}, Purpose=rating_modification",
                operation, firebaseUid, opportunityId, rating);

        try {
            CoopDto result = serviceCall.apply(opportunityId, rating);
            MappingJacksonValue mapping = new MappingJacksonValue(result);
            mapping.setSerializationView(viewClass);

            log.info("GDPR: UPDATE_COMPLETE Operation={}, FirebaseUID={}, OpportunityID={}, Success=true",
                    operation, firebaseUid, opportunityId);
            return ResponseEntity.ok(mapping);
        } catch (Exception e) {
            log.error("GDPR: Operation={}_FAILED, FirebaseUID={}, OpportunityID={}, Rating={}, Error={}",
                    operation, firebaseUid, opportunityId, rating, e.getMessage(), e);
            throw e;
        }
    }

    private Pageable pageable(int page, Integer size, Sort.Direction direction) {
        int pageSize = clampedPageSize(size);
        return PageRequest.of(page, pageSize, Sort.by(direction, SORT_FIELD));
    }

    private int clampedPageSize(Integer requested) {
        int pageSize = requested != null ? requested : defaultPageSize;
        return Math.min(pageSize, MAX_PAGE_SIZE);
    }

    private String currentFirebaseUid() {
        return SecurityContextHolder.getContext()
                .getAuthentication()
                .getPrincipal()
                .toString();
    }
}
