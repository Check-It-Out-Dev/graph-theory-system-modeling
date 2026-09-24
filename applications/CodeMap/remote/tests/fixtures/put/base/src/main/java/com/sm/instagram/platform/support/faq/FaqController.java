package com.sm.instagram.platform.support.faq;

import com.sm.instagram.platform.common.base.BaseController;
import com.sm.instagram.platform.common.base.BaseService;
import com.sm.instagram.platform.common.exceptions.ResourceNotFoundException;
import com.sm.instagram.platform.common.ratelimit.RateLimit;
import com.sm.instagram.platform.common.ratelimit.RateLimitKeyType;
import com.sm.instagram.platform.common.ratelimit.RateLimitProfile;
import com.sm.instagram.platform.support.faq.dtos.FaqDtoIn;
import com.sm.instagram.platform.support.faq.dtos.FaqDtoOut;
import com.sm.instagram.platform.support.faq.models.Faq;
import com.sm.instagram.platform.support.faq.services.FaqService;
import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * REST controller for FAQ operations.
 */
@Slf4j
@RestController
@RequestMapping("/support/faq")
@RateLimit(profile = RateLimitProfile.STANDARD, keyType = RateLimitKeyType.USER_ENDPOINT)
public class FaqController extends BaseController<Faq, Long, FaqDtoIn, FaqDtoOut> {
    private final FaqService faqService;

    public FaqController(FaqService faqService) {
        super(Faq.class);
        this.faqService = faqService;
    }

    @Override
    protected BaseService<Faq, Long, FaqDtoIn> getService() {
        return faqService;
    }

    /**
     * Get all active FAQs.
     *
     * @return List of active FAQs
     */
    @GetMapping("/active")
    public ResponseEntity<List<FaqDtoOut>> getAllActiveFaqs() {
        String firebaseUid = getAuthenticatedAdminUid();


        log.info("GDPR: Operation=getAllActiveFaqs, FirebaseUID={}, DataAccessed=faq.questions,faq.answers, Purpose=user_support",
                firebaseUid);

        List<FaqDtoOut> dtos = faqService.getAllActiveFaqsAsDto();
        return ResponseEntity.ok(dtos);
    }

    /**
     * Get FAQs for a specific category.
     *
     * @param categoryId Category ID
     * @return List of FAQs in the category
     */
    @GetMapping("/by-category/{categoryId}")
    public ResponseEntity<List<FaqDtoOut>> getFaqsByCategory(@PathVariable Long categoryId) {
        String firebaseUid = getAuthenticatedAdminUid();


        log.info("GDPR: Operation=getFaqsByCategory, FirebaseUID={}, CategoryID={}, DataAccessed=faq.questions,faq.answers, Purpose=user_support",
                firebaseUid, categoryId);

        List<FaqDtoOut> dtos = faqService.getFaqsByCategoryAsDto(categoryId);
        return ResponseEntity.ok(dtos);
    }

    /**
     * Search FAQs.
     *
     * @param query Search term
     * @return List of matching FAQs
     */
    @GetMapping("/search")
    public ResponseEntity<List<FaqDtoOut>> searchFaqs(@RequestParam(required = false) String query) {
        String firebaseUid = getAuthenticatedAdminUid();


        log.info("GDPR: Operation=searchFaqs, FirebaseUID={}, Query={}, DataAccessed=faq.questions,faq.answers, Purpose=user_search",
                firebaseUid, query);

        List<FaqDtoOut> dtos = faqService.searchFaqsAsDto(query);
        return ResponseEntity.ok(dtos);
    }

    /**
     * Create a new FAQ.
     *
     * @param dto FAQ data
     * @return Created FAQ
     */
    @PostMapping
    @PreAuthorize("hasAuthority('ADMIN')")
    @Override
    public ResponseEntity<FaqDtoOut> create(@Valid @RequestBody FaqDtoIn dto) {
        String firebaseUid = getAuthenticatedAdminUid();


        log.info("GDPR: Operation=createFaq, FirebaseUID={}, Action=CREATE, DataCreated=faq.question,faq.answer, Purpose=admin_management",
                firebaseUid);

        FaqDtoOut faqDto = faqService.createFaqAsDto(dto);

        log.info("GDPR: Operation=createFaq_complete, FirebaseUID={}, FaqID={}, Purpose=admin_management",
                firebaseUid, faqDto.getId());

        return ResponseEntity.status(HttpStatus.CREATED)
                .body(faqDto);
    }

    /**
     * Update an existing FAQ.
     *
     * @param id  FAQ ID
     * @param dto Updated FAQ data
     * @return Updated FAQ
     */
    @Override
    @PutMapping("/{id}")
    @PreAuthorize("hasAuthority('ADMIN')")
    public ResponseEntity<FaqDtoOut> update(@PathVariable Long id, @Valid @RequestBody FaqDtoIn dto) {
        FaqDtoOut faqDto = faqService.updateAsDto(id, dto);
        return ResponseEntity.ok(faqDto);
    }

    /**
     * Get a specific FAQ by ID.
     *
     * @param id FAQ ID
     * @return FAQ data
     */
    @GetMapping("/{id}")
    @Override
    public ResponseEntity<FaqDtoOut> getById(@PathVariable Long id) {
        FaqDtoOut faqDto = faqService.findByIdAsDto(id);
        return ResponseEntity.ok(faqDto);
    }

    /**
     * Soft delete an FAQ by setting active to false.
     *
     * @param id FAQ ID
     * @return No content response
     */
    @DeleteMapping("/{id}/soft")
    @PreAuthorize("hasAuthority('ADMIN')")
    public ResponseEntity<Void> softDelete(@PathVariable Long id) {
        String firebaseUid = getAuthenticatedAdminUid();


        log.warn("GDPR: SOFT_DELETION Operation=softDeleteFaq, FirebaseUID={}, FaqID={}, Purpose=admin_management",
                firebaseUid, id);

        faqService.softDelete(id);

        log.info("GDPR: SOFT_DELETION_COMPLETE FaqID={}, FirebaseUID={}, Active=false",
                id, firebaseUid);

        return ResponseEntity.noContent().build();
    }

    /**
     * Update display order of an FAQ.
     *
     * @param id    FAQ ID
     * @param order New display order
     * @return Updated FAQ
     */
    @PatchMapping("/{id}/display-order/{order}")
    @PreAuthorize("hasAuthority('ADMIN')")
    public ResponseEntity<FaqDtoOut> updateDisplayOrder(
            @PathVariable Long id,
            @PathVariable int order) {
        FaqDtoOut faqDto = faqService.updateDisplayOrderAsDto(id, order);
        return ResponseEntity.ok(faqDto);
    }

    @Override
    @PatchMapping("/{id}")
    @PreAuthorize("hasAuthority('ADMIN')")
    public ResponseEntity<FaqDtoOut> patch(@PathVariable Long id, @RequestBody Map<String, Object> updates) {
        return super.patch(id, updates);
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
        if (auth == null || auth.getPrincipal() == null
                || "anonymousUser".equals(auth.getPrincipal())) {
            return "anonymous";
        }
        return auth.getPrincipal().toString();
    }
}
