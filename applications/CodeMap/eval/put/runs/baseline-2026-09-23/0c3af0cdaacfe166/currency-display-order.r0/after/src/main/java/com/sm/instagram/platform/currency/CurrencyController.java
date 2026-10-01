package com.sm.instagram.platform.currency;

import com.sm.instagram.platform.common.base.BaseController;
import com.sm.instagram.platform.common.base.BaseService;
import com.sm.instagram.platform.common.exceptions.AuthenticationTranslatableException;
import com.sm.instagram.platform.common.exceptions.ResourceNotFoundException;
import com.sm.instagram.platform.common.exceptions.ValidationTranslatableException;
import com.sm.instagram.platform.common.ratelimit.RateLimit;
import com.sm.instagram.platform.common.ratelimit.RateLimitKeyType;
import com.sm.instagram.platform.common.ratelimit.RateLimitProfile;
import com.sm.instagram.platform.common.translation.TranslationService;
import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.Locale;
import java.util.Map;

/**
 * REST controller for managing currencies.
 * Provides endpoints for retrieving currency data with localized names.
 * Requires admin authentication for all operations.
 */
@Slf4j
@RestController
@RequestMapping("currency")
@PreAuthorize("hasAuthority('ADMIN')")
@RateLimit(profile = RateLimitProfile.STANDARD, keyType = RateLimitKeyType.USER_ENDPOINT)
public class CurrencyController extends BaseController<Currency, Long, CurrencyDto, CurrencyDtoOut> {

    private final CurrencyService currencyService;
    private final TranslationService translationService;
    private final HttpServletRequest request;

    protected CurrencyController(CurrencyService currencyService,
                                 TranslationService translationService, HttpServletRequest request) {
        super(Currency.class);
        this.currencyService = currencyService;
        this.translationService = translationService;
        this.request = request;
    }

    /**
     * Get locale from Accept-Language header
     *
     * @return Locale parsed from Accept-Language header or default Polish locale
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
     * Get currency by ID with localized name.
     *
     * @param id The currency ID
     * @return Currency data with translated name
     * @throws ValidationTranslatableException if ID is invalid
     * @throws ResourceNotFoundException       if authentication missing or currency not found
     */
    @Override
    @PreAuthorize("isAuthenticated()")
    @GetMapping("/{id}")
    public ResponseEntity<CurrencyDtoOut> getById(@PathVariable Long id) {
        // Validate ID parameter
        if (id == null || id <= 0) {
            throw new ValidationTranslatableException("error.validation.invalid_id", "id");
        }

        String firebaseUid = getAuthenticatedAdminUid();
        log.info("GDPR: Operation=getCurrencyById, FirebaseUID={}, CurrencyID={}, Purpose=reference_data_retrieval",
                firebaseUid, id);
        log.info("Retrieving currency with ID: {}", id);
        long startTime = System.currentTimeMillis();

        Locale locale = getLocaleFromRequest();
        Currency entity = currencyService.findById(id);

        // Convert to translation DTO with manual mapping
        CurrencyDtoOut dto = CurrencyDtoOut.builder()
                .id(entity.getId())
                .originalName(entity.getName())
                .name(translationService.translateCurrency(entity.getIsoCode(), locale))
                .isoCode(entity.getIsoCode())
                .sign(entity.getSign())
                .countryCode(entity.getCountryCode())
                .displayOrder(entity.getDisplayOrder())
                .build();

        long duration = System.currentTimeMillis() - startTime;
        log.info("GDPR: DataAccessed=currency.name,currency.isoCode,currency.sign, FirebaseUID={}, CurrencyID={}, Purpose=display",
                firebaseUid, id);
        log.info("Successfully retrieved currency with ID: {} in {}ms", id, duration);

        return ResponseEntity.ok(dto);
    }

    /**
     * Get paginated currencies with localized names.
     *
     * @param pageable Pagination parameters
     * @param filters  Filter parameters
     * @return Page of currencies with translated names
     * @throws ValidationTranslatableException if pagination parameters are invalid
     * @throws ResourceNotFoundException       if authentication missing
     */
    @Override
    @PreAuthorize("isAuthenticated()")
    @GetMapping(value = "/paged")
    public ResponseEntity<Page<CurrencyDtoOut>> findPaginated(Pageable pageable, @RequestParam Map<String, String> filters) {
        // Validate pagination parameters
        if (pageable.getPageSize() > 1000) {
            throw new ValidationTranslatableException("error.validation.list_too_large", "1000");
        }
        if (pageable.getPageNumber() < 0) {
            throw new ValidationTranslatableException("error.validation.invalid_parameter", "page number must be non-negative");
        }

        String firebaseUid = getAuthenticatedAdminUid();
        log.info("GDPR: Operation=getCurrenciesPaged, FirebaseUID={}, PageNumber={}, PageSize={}, Purpose=reference_data_listing",
                firebaseUid, pageable.getPageNumber(), pageable.getPageSize());
        log.info("Retrieving paginated currencies - page: {}, size: {}, filters: {}",
                pageable.getPageNumber(), pageable.getPageSize(), filters.keySet());
        long startTime = System.currentTimeMillis();

        Locale locale = getLocaleFromRequest();

        // Remove pagination parameters from filters
        filters.remove("page");
        filters.remove("size");
        filters.remove("sort");
        filters.remove("direction");

        Page<Currency> page = currencyService.getDataPagedAndFiltered(pageable, filters);

        // Convert to translation DTOs with manual mapping
        Page<CurrencyDtoOut> dtoPage = page.map(entity ->
                CurrencyDtoOut.builder()
                        .id(entity.getId())
                        .originalName(entity.getName())
                        .name(translationService.translateCurrency(entity.getIsoCode(), locale))
                        .isoCode(entity.getIsoCode())
                        .sign(entity.getSign())
                        .countryCode(entity.getCountryCode())
                        .displayOrder(entity.getDisplayOrder())
                        .build()
        );

        long duration = System.currentTimeMillis() - startTime;
        log.info("GDPR: DataAccessed=currencies, FirebaseUID={}, RecordCount={}, TotalElements={}, Purpose=reference_listing",
                firebaseUid, dtoPage.getNumberOfElements(), dtoPage.getTotalElements());
        log.info("Successfully retrieved {} currencies (total: {}) in {}ms",
                dtoPage.getNumberOfElements(), dtoPage.getTotalElements(), duration);
        log.debug("Applied filters: {}", filters);

        return ResponseEntity.ok(dtoPage);
    }

    @Override
    protected BaseService<Currency, Long, CurrencyDto> getService() {
        return currencyService;
    }

    // ===== HELPER METHODS =====

    /**
     * Gets the authenticated admin UID from the security context.
     *
     * @return The Firebase UID of the authenticated admin user
     * @throws ResourceNotFoundException if authentication context is missing
     */
    private String getAuthenticatedAdminUid() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || auth.getPrincipal() == null) {
            log.error("Authentication not available in security context");
            throw new AuthenticationTranslatableException("error.auth.not_authenticated");
        }
        return auth.getPrincipal().toString();
    }
}
