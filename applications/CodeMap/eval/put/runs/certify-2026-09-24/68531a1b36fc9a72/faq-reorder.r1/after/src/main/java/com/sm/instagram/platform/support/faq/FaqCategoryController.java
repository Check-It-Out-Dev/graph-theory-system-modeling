package com.sm.instagram.platform.support.faq;

import com.sm.instagram.platform.common.base.BaseController;
import com.sm.instagram.platform.common.base.BaseService;
import com.sm.instagram.platform.common.ratelimit.RateLimit;
import com.sm.instagram.platform.common.ratelimit.RateLimitKeyType;
import com.sm.instagram.platform.common.ratelimit.RateLimitProfile;
import com.sm.instagram.platform.support.faq.dtos.FaqCategoryDtoIn;
import com.sm.instagram.platform.support.faq.dtos.FaqCategoryDtoOut;
import com.sm.instagram.platform.support.faq.dtos.FaqReorderDtoIn;
import com.sm.instagram.platform.support.faq.models.FaqCategory;
import com.sm.instagram.platform.support.faq.services.FaqCategoryService;
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
 * REST controller for FAQ Category operations.
 */
@Slf4j
@RestController
@RequestMapping("/support/faq/categories")
@RateLimit(profile = RateLimitProfile.STANDARD, keyType = RateLimitKeyType.USER_ENDPOINT)
public class FaqCategoryController extends BaseController<FaqCategory, Long, FaqCategoryDtoIn, FaqCategoryDtoOut> {
    private final FaqCategoryService faqCategoryService;
    private final FaqService faqService;

    public FaqCategoryController(FaqCategoryService faqCategoryService, FaqService faqService) {
        super(FaqCategory.class);
        this.faqCategoryService = faqCategoryService;
        this.faqService = faqService;
    }

    @Override
    protected BaseService<FaqCategory, Long, FaqCategoryDtoIn> getService() {
        return faqCategoryService;
    }

    /**
     * Get all active FAQ categories.
     *
     * @return List of active categories
     */
    @GetMapping("/active")
    public ResponseEntity<List<FaqCategoryDtoOut>> getAllActiveCategories() {
        String firebaseUid = extractFirebaseUid();
        log.info("GDPR: Operation=getAllActiveCategories, FirebaseUID={}, Purpose=faq_retrieval", firebaseUid);

        List<FaqCategoryDtoOut> dtos = faqCategoryService.getAllActiveCategoriesAsDto();

        log.info("GDPR: DataAccessed=faq_categories, FirebaseUID={}, RecordCount={}, Purpose=support_content",
                firebaseUid, dtos.size());
        return ResponseEntity.ok(dtos);
    }

    /**
     * Create a new FAQ category.
     * Uses the base controller's create method instead of defining a duplicate.
     */
    @Override
    @PostMapping
    @PreAuthorize("hasAuthority('ADMIN')")
    public ResponseEntity<FaqCategoryDtoOut> create(@Valid @RequestBody FaqCategoryDtoIn dto) {
        String firebaseUid = extractFirebaseUid();
        log.info("GDPR: Operation=createFaqCategory, FirebaseUID={}, CategoryName={}, Purpose=content_management",
                firebaseUid, dto.getName());

        FaqCategoryDtoOut categoryDto = faqCategoryService.createCategoryAsDto(dto);

        log.info("GDPR: DataCreated=faq_category, FirebaseUID={}, CategoryID={}, Purpose=admin_content_creation",
                firebaseUid, categoryDto.getId());
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(categoryDto);
    }

    /**
     * Update an existing FAQ category.
     *
     * @param id  Category ID
     * @param dto Updated category data
     * @return Updated category
     */
    @PutMapping("/{id}")
    @PreAuthorize("hasAuthority('ADMIN')")
    @Override
    public ResponseEntity<FaqCategoryDtoOut> update(@PathVariable Long id, @Valid @RequestBody FaqCategoryDtoIn dto) {
        String firebaseUid = extractFirebaseUid();
        log.info("GDPR: Operation=updateFaqCategory, FirebaseUID={}, CategoryID={}, Purpose=content_update",
                firebaseUid, id);

        ResponseEntity<FaqCategoryDtoOut> response = super.update(id, dto);

        log.info("GDPR: DataModified=faq_category, FirebaseUID={}, CategoryID={}, Purpose=admin_content_update",
                firebaseUid, id);
        return response;
    }

    /**
     * Get a specific FAQ category by ID.
     *
     * @param id Category ID
     * @return Category data
     */
    @GetMapping("/{id}")
    @Override
    public ResponseEntity<FaqCategoryDtoOut> getById(@PathVariable Long id) {
        String firebaseUid = extractFirebaseUid();
        log.info("GDPR: Operation=getFaqCategoryById, FirebaseUID={}, CategoryID={}, Purpose=content_retrieval",
                firebaseUid, id);

        FaqCategoryDtoOut categoryDto = faqCategoryService.findByIdAsDto(id);

        log.info("GDPR: DataAccessed=faq_category_details, FirebaseUID={}, CategoryID={}, Purpose=support_content",
                firebaseUid, id);
        return ResponseEntity.ok(categoryDto);
    }

    /**
     * Soft delete a category by setting active to false.
     *
     * @param id Category ID
     * @return No content response
     */
    @DeleteMapping("/{id}/soft")
    @PreAuthorize("hasAuthority('ADMIN')")
    public ResponseEntity<Void> softDelete(@PathVariable Long id) {
        String firebaseUid = extractFirebaseUid();
        log.warn("GDPR: SOFT_DELETION Operation=softDeleteFaqCategory, FirebaseUID={}, CategoryID={}, Purpose=content_archival",
                firebaseUid, id);

        faqCategoryService.softDelete(id);

        log.info("GDPR: SOFT_DELETION_COMPLETE CategoryID={}, FirebaseUID={}, Active=false", id, firebaseUid);
        return ResponseEntity.noContent().build();
    }

    /**
     * Update display order of a category.
     *
     * @param id    Category ID
     * @param order New display order
     * @return Updated category
     */
    @PatchMapping("/{id}/display-order/{order}")
    @PreAuthorize("hasAuthority('ADMIN')")
    public ResponseEntity<FaqCategoryDtoOut> updateDisplayOrder(
            @PathVariable Long id,
            @PathVariable int order) {
        String firebaseUid = extractFirebaseUid();
        log.info("GDPR: Operation=updateFaqCategoryDisplayOrder, FirebaseUID={}, CategoryID={}, NewOrder={}, Purpose=content_organization",
                firebaseUid, id, order);

        FaqCategoryDtoOut categoryDto = faqCategoryService.updateDisplayOrderAsDto(id, order);

        log.info("GDPR: DataModified=faq_category_order, FirebaseUID={}, CategoryID={}, Purpose=admin_content_ordering",
                firebaseUid, id);
        return ResponseEntity.ok(categoryDto);
    }

    @Override
    @PatchMapping("/{id}")
    @PreAuthorize("hasAuthority('ADMIN')")
    public ResponseEntity<FaqCategoryDtoOut> patch(@PathVariable Long id, @RequestBody Map<String, Object> updates) {
        return super.patch(id, updates);
    }

    /**
     * Reorder the FAQs of a category in one request.
     *
     * @param categoryId Category ID
     * @param dto        The active FAQ ids of the category, in the desired display order
     * @return No content response
     */
    @PutMapping("/{categoryId}/order")
    @PreAuthorize("hasAuthority('ADMIN')")
    public ResponseEntity<Void> reorderFaqs(@PathVariable Long categoryId, @Valid @RequestBody FaqReorderDtoIn dto) {
        String firebaseUid = extractFirebaseUid();
        log.info("GDPR: Operation=reorderFaqs, FirebaseUID={}, CategoryID={}, Purpose=admin_content_ordering",
                firebaseUid, categoryId);

        faqService.reorderCategory(categoryId, dto.getFaqIds());

        log.info("GDPR: DataModified=faq_order, FirebaseUID={}, CategoryID={}, Purpose=admin_content_ordering",
                firebaseUid, categoryId);
        return ResponseEntity.noContent().build();
    }

    private String extractFirebaseUid() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || auth.getPrincipal() == null
                || "anonymousUser".equals(auth.getPrincipal())) {
            return "anonymous";
        }
        return auth.getPrincipal().toString();
    }
}