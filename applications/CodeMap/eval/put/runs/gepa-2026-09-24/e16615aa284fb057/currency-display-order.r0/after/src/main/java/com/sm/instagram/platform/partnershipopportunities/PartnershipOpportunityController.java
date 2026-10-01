package com.sm.instagram.platform.partnershipopportunities;

import com.sm.instagram.platform.common.base.BaseController;
import com.sm.instagram.platform.common.base.BaseService;
import com.sm.instagram.platform.common.ratelimit.RateLimit;
import com.sm.instagram.platform.common.ratelimit.RateLimitKeyType;
import com.sm.instagram.platform.common.ratelimit.RateLimitProfile;
import com.sm.instagram.platform.common.translation.TranslationService;
import com.sm.instagram.platform.currency.CurrencyDtoOut;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;

import org.springframework.data.domain.Page;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

@Slf4j
@RestController
@PreAuthorize("hasAuthority('ADMIN') or hasAuthority('COMPANY') or hasAuthority('INFLUENCER')")
@RequestMapping("partnership-opportunity")
@Tag(name = "Partnership Opportunity API", description = "Endpoints for managing Partnership Opportunity data")
@RateLimit(profile = RateLimitProfile.STANDARD, keyType = RateLimitKeyType.USER_ENDPOINT)  // 60 req/min per endpoint
public class PartnershipOpportunityController extends BaseController<PartnershipOpportunity, Long, PartnershipOpportunityDtoIn, PartnershipOpportunityDtoOut> {

    private final PartnershipOpportunityService opportunitiesService;
    private final TranslationService translationService;
    private final HttpServletRequest request;

    public PartnershipOpportunityController(PartnershipOpportunityService opportunitiesService, 
                                          TranslationService translationService,
                                          HttpServletRequest request) {
        super(PartnershipOpportunity.class);
        this.opportunitiesService = opportunitiesService;
        this.translationService = translationService;
        this.request = request;
    }

    @Override
    protected BaseService<PartnershipOpportunity, Long, PartnershipOpportunityDtoIn> getService() {
        return opportunitiesService;
    }

    /**
     * Get locale from Accept-Language header
     */
    private Locale getLocaleFromRequest() {
        String acceptLanguageHeader = request.getHeader("Accept-Language");
        if (acceptLanguageHeader != null && !acceptLanguageHeader.isEmpty()) {
            // Parse the first language from Accept-Language header
            String language = acceptLanguageHeader.split(",")[0].split(";")[0].trim();
            return Locale.forLanguageTag(language);
        }
        return Locale.forLanguageTag("pl"); // Default to Polish
    }

    /**
     * Translate CompensationType enum to DTO
     */
    private CompensationTypeDtoOut translateCompensationType(CompensationType compensationType, Locale locale) {
        if (compensationType == null) {
            return null;
        }
        
        String translatedLabel = translationService.translateCompensationType(compensationType.name(), locale);
        
        return CompensationTypeDtoOut.builder()
                .value(compensationType.name())
                .label(translatedLabel)
                .originalLabel(compensationType.name())
                .build();
    }

    @Override
    @PostMapping
    @Operation(summary = "Create a new opportunity", description = "Creates a new opportunity based on the provided data")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Partnership opportunity created successfully",
                    content = @Content(schema = @Schema(implementation = PartnershipOpportunityDtoOut.class))),
            @ApiResponse(responseCode = "400", description = "Invalid input data"),
            @ApiResponse(responseCode = "403", description = "User not authorized to create this opportunity")
    })
    public ResponseEntity<PartnershipOpportunityDtoOut> create(@Valid @RequestBody PartnershipOpportunityDtoIn dto) {
        String firebaseUid = getFirebaseUid();
        log.info("GDPR: Operation=createPartnershipOpportunity, FirebaseUID={}, CompanyID={}, Purpose=business_operation", 
            firebaseUid, dto.getCompany());
        log.debug("Creating partnership opportunity with data: {}", dto);
        long startTime = System.currentTimeMillis();

        Locale locale = getLocaleFromRequest();
        PartnershipOpportunityDtoOut result = opportunitiesService.saveFromDtoAsDto(dto, locale);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully created partnership opportunity with ID: {} in {}ms",
                result.getId(), duration);

        return ResponseEntity.ok(result);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    @GetMapping("/{id}")
    @Operation(summary = "Get partnership opportunity by ID", description = "Retrieves a partnership opportunity by its unique identifier")
    @PreAuthorize("isAuthenticated()")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Partnership opportunity found",
                    content = @Content(schema = @Schema(implementation = PartnershipOpportunityDtoOut.class))),
            @ApiResponse(responseCode = "404", description = "Partnership opportunity not found"),
            @ApiResponse(responseCode = "403", description = "User not authorized to access this partnership opportunity")
    })
    public ResponseEntity<PartnershipOpportunityDtoOut> getById(@Parameter(description = "Partnership opportunity ID") @PathVariable Long id) {
        String firebaseUid = getFirebaseUid();
        log.info("GDPR: Operation=getPartnershipOpportunity, FirebaseUID={}, OpportunityID={}, Purpose=data_retrieval", 
            firebaseUid, id);
        long startTime = System.currentTimeMillis();

        Locale locale = getLocaleFromRequest();
        PartnershipOpportunityDtoOut result = opportunitiesService.findByIdAsDto(id, locale);

        // Translate enum fields in the DTO
        if (result.getCompensationType() != null) {
            // Find the original entity to get the enum value
            PartnershipOpportunity entity = opportunitiesService.findById(id);
            CompensationTypeDtoOut translatedCompensationType = translateCompensationType(entity.getCompensationType(), locale);
            result.setCompensationType(translatedCompensationType);
        }

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully retrieved partnership opportunity with ID: {} in {}ms", id, duration);
        log.debug("Retrieved partnership opportunity: {} - {}", result.getTitle(), result.getCompensationType());

        return ResponseEntity.ok(result);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    @GetMapping("/paged")
    @RateLimit(profile = RateLimitProfile.RELAXED)  // 120 req/min for browsing opportunities
    @PreAuthorize("isAuthenticated()")
    @Operation(summary = "Get paginated partnership opportunities",
            description = "Retrieves a paginated list of partnership opportunities with optional filtering")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Paginated list retrieved successfully",
                    content = @Content(schema = @Schema(implementation = Page.class))),
            @ApiResponse(responseCode = "400", description = "Invalid filter or pagination parameters"),
            @ApiResponse(responseCode = "403", description = "User not authorized to access partnership opportunities")
    })
    public ResponseEntity<Page<PartnershipOpportunityDtoOut>> findPaginated(
            @Parameter(description = "Pagination parameters") Pageable pageable,
            @Parameter(description = "Filter parameters") @RequestParam Map<String, String> filters) {
        
        String firebaseUid = getFirebaseUid();
        log.info("GDPR: Operation=listPartnershipOpportunities, FirebaseUID={}, Page={}, Size={}, Filters={}, Purpose=data_browsing",
                firebaseUid, pageable.getPageNumber(), pageable.getPageSize(), filters.keySet());
        long startTime = System.currentTimeMillis();

        Locale locale = getLocaleFromRequest();

        // Remove pagination parameters from filters
        filters.remove("page");
        filters.remove("size");
        filters.remove("sort");
        filters.remove("direction");

        Page<PartnershipOpportunityDtoOut> dtoPage = opportunitiesService.getDataPagedAndFilteredAsDtos(pageable, filters, locale);

        // Translate enum fields for each DTO in the page
        Page<PartnershipOpportunity> entityPage = opportunitiesService.getDataPagedAndFiltered(pageable, filters);
        Page<PartnershipOpportunityDtoOut> translatedDtoPage = dtoPage.map(dto -> {
            // Find the corresponding entity to get enum values
            PartnershipOpportunity entity = entityPage.getContent().stream()
                    .filter(e -> e.getId().equals(dto.getId()))
                    .findFirst()
                    .orElse(null);
            
            if (entity != null && entity.getCompensationType() != null) {
                CompensationTypeDtoOut translatedCompensationType = translateCompensationType(entity.getCompensationType(), locale);
                dto.setCompensationType(translatedCompensationType);
            }

            // Contract gap found by the FE rewrite: the list DTOs shipped
            // without currency (the detail path maps it), so clients could
            // not label CASH amounts. Patch it here exactly like the detail
            // path does - the entity is already in hand.
            if (entity != null && entity.getCurrency() != null && dto.getCurrency() == null) {
                dto.setCurrency(CurrencyDtoOut.builder()
                        .id(entity.getCurrency().getId())
                        .originalName(entity.getCurrency().getName())
                        .name(translationService.translateCurrency(entity.getCurrency().getIsoCode(), locale))
                        .isoCode(entity.getCurrency().getIsoCode())
                        .sign(entity.getCurrency().getSign())
                        .countryCode(entity.getCurrency().getCountryCode())
                        .displayOrder(entity.getCurrency().getDisplayOrder())
                        .build());
            }

            return dto;
        });

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully retrieved {} partnership opportunities (total: {}) in {}ms",
                translatedDtoPage.getNumberOfElements(), translatedDtoPage.getTotalElements(), duration);

        return ResponseEntity.ok(translatedDtoPage);
    }

    /**
     * Retrieves all available compensation types
     *
     * @return list of compensation type names
     */
    @GetMapping(value = "/compensation/type", produces = "application/json")
    @PreAuthorize("isAuthenticated()")
    @Operation(summary = "Get compensation types", description = "Retrieves all available compensation types")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Compensation types retrieved successfully"),
            @ApiResponse(responseCode = "403", description = "User not authorized to access compensation types")
    })
    public ResponseEntity<List<String>> getCompensationTypeList() {
        String firebaseUid = getFirebaseUid();
        log.info("GDPR: Operation=getCompensationTypes, FirebaseUID={}, Purpose=metadata_retrieval", firebaseUid);

        List<String> compensationTypes = Arrays.stream(CompensationType.values())
                .map(Enum::name)
                .toList();

        log.debug("Retrieved {} compensation types: {}", compensationTypes.size(), compensationTypes);
        return ResponseEntity.ok(compensationTypes);
    }

    /**
     * Updates an existing partnership opportunity
     */
    @Override
    @PutMapping("/{id}")
    @RateLimit(profile = RateLimitProfile.STANDARD, keyType = RateLimitKeyType.USER_ENDPOINT, skipForAdmin = true)  // 60 req/min per endpoint, skip for admin
    @Operation(summary = "Update partnership opportunity", description = "Updates an existing partnership opportunity with the provided data")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Partnership opportunity updated successfully"),
            @ApiResponse(responseCode = "400", description = "Invalid input data"),
            @ApiResponse(responseCode = "403", description = "User not authorized to update partnership opportunity"),
            @ApiResponse(responseCode = "404", description = "Partnership opportunity not found")
    })
    public ResponseEntity<PartnershipOpportunityDtoOut> update(@PathVariable Long id, @Valid @RequestBody PartnershipOpportunityDtoIn dto) {
        String firebaseUid = getFirebaseUid();
        log.info("GDPR: Operation=updatePartnershipOpportunity, FirebaseUID={}, OpportunityID={}, Purpose=data_modification", 
            firebaseUid, id);
        long startTime = System.currentTimeMillis();

        Locale locale = getLocaleFromRequest();
        PartnershipOpportunityDtoOut result = opportunitiesService.updateAsDto(id, dto, locale);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully updated partnership opportunity with ID: {} in {}ms", id, duration);

        return ResponseEntity.ok(result);
    }

    /**
     * Partially updates a partnership opportunity
     */
    @Override
    @PatchMapping("/{id}")
    @RateLimit(profile = RateLimitProfile.STANDARD, keyType = RateLimitKeyType.USER_ENDPOINT, skipForAdmin = true)  // 60 req/min per endpoint, skip for admin
    @Operation(summary = "Partially update partnership opportunity", description = "Updates specific fields of an existing partnership opportunity")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Partnership opportunity updated successfully"),
            @ApiResponse(responseCode = "400", description = "Invalid input data"),
            @ApiResponse(responseCode = "403", description = "User not authorized to update partnership opportunity"),
            @ApiResponse(responseCode = "404", description = "Partnership opportunity not found")
    })
    public ResponseEntity<PartnershipOpportunityDtoOut> patch(@PathVariable Long id, @Valid @RequestBody Map<String, Object> updates) {
        String firebaseUid = getFirebaseUid();
        log.info("GDPR: Operation=patchPartnershipOpportunity, FirebaseUID={}, OpportunityID={}, UpdatedFields={}, Purpose=partial_update", 
            firebaseUid, id, updates.keySet());
        long startTime = System.currentTimeMillis();

        Locale locale = getLocaleFromRequest();
        PartnershipOpportunityDtoOut result = opportunitiesService.patchAsDto(id, updates, locale);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully patched partnership opportunity with ID: {} in {}ms", id, duration);

        return ResponseEntity.ok(result);
    }
    
    /**
     * Extract Firebase UID from security context
     */
    private String getFirebaseUid() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth != null && auth.isAuthenticated() && auth.getPrincipal() != null) {
            return auth.getPrincipal().toString();
        }
        return "anonymous";
    }
}
