package com.sm.instagram.platform.support.faq.services;

import com.sm.instagram.platform.common.base.BaseService;
import com.sm.instagram.platform.common.exceptions.ResourceNotFoundException;
import com.sm.instagram.platform.common.util.RepositoryResolver;
import com.sm.instagram.platform.common.util.filtering.SpecificationBuilder;
import com.sm.instagram.platform.support.faq.dtos.FaqDtoIn;
import com.sm.instagram.platform.support.faq.dtos.FaqDtoOut;
import com.sm.instagram.platform.support.faq.models.Faq;
import com.sm.instagram.platform.support.faq.models.FaqCategory;
import com.sm.instagram.platform.support.faq.repositories.FaqCategoryRepository;
import com.sm.instagram.platform.support.faq.repositories.FaqRepository;
import lombok.extern.slf4j.Slf4j;
import org.modelmapper.ModelMapper;
import org.springframework.context.ApplicationContext;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Service for managing FAQ entries.
 */
@Slf4j
@Service
@Transactional
@SuppressWarnings("unchecked")
public class FaqService extends BaseService<Faq, Long, FaqDtoIn> {
    private final FaqRepository faqRepository;
    private final FaqCategoryRepository faqCategoryRepository;

    public FaqService(
            SpecificationBuilder<Faq> specificationBuilder,
            FaqRepository faqRepository,
            FaqCategoryRepository faqCategoryRepository,
            ModelMapper modelMapper,
            RepositoryResolver repositoryResolver,
            ApplicationContext applicationContext) {
        super(applicationContext, specificationBuilder, faqRepository, modelMapper, repositoryResolver);
        this.faqRepository = faqRepository;
        this.faqCategoryRepository = faqCategoryRepository;
    }

    /**
     * Get all active FAQs ordered by display order.
     *
     * @return List of active FAQs
     */
    public List<Faq> getAllActiveFaqs() {
        log.debug("GDPR: Service=getAllActiveFaqs, DataAccessed=faq.all_active, Purpose=data_retrieval");
        return faqRepository.findByActiveTrueOrderByDisplayOrderAsc();
    }

    /**
     * Get all active FAQs for a specific category.
     *
     * @param categoryId The category ID
     * @return List of active FAQs in the category
     */
    public List<Faq> getFaqsByCategory(Long categoryId) {
        FaqCategory category = faqCategoryRepository.findById(categoryId)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "FAQ Category"));
        return faqRepository.findByCategoryAndActiveTrueOrderByDisplayOrderAsc(category);
    }

    /**
     * Search FAQs by a query term.
     *
     * @param query The search term
     * @return List of matching FAQs
     */
    public List<Faq> searchFaqs(String query) {
        if (query == null || query.trim().isEmpty()) {
            return getAllActiveFaqs();
        }
        return faqRepository.search(query.trim());
    }

    /**
     * Create a new FAQ.
     *
     * @param dto The FAQ data
     * @return The created FAQ
     */
    @Transactional
    public Faq createFaq(FaqDtoIn dto) {
        log.info("GDPR: Service=createFaq, Operation=CREATE, DataFields=question,answer,category, Purpose=content_management");

        Faq faq = modelMapper.map(dto, Faq.class);

        // Get the category
        FaqCategory category = faqCategoryRepository.findById(dto.getCategoryId())
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "FAQ Category"));
        faq.setCategory(category);

        // If display order is 0, set it to the next available order in this category
        if (faq.getDisplayOrder() == 0) {
            int maxOrder = faqRepository.getMaxDisplayOrder(dto.getCategoryId());
            faq.setDisplayOrder(maxOrder + 1);
        }

        return save(faq);
    }

    /**
     * Update an existing FAQ.
     *
     * @param id  The FAQ ID
     * @param dto The updated FAQ data
     * @return The updated FAQ
     */
    @Transactional
    @Override
    public Faq update(Long id, FaqDtoIn dto) {
        Faq faq = findById(id);

        // Update simple properties
        faq.setQuestion(dto.getQuestion());
        faq.setAnswer(dto.getAnswer());
        faq.setDisplayOrder(dto.getDisplayOrder());
        faq.setActive(dto.isActive());

        // Update category if it has changed
        if (!faq.getCategory().getId().equals(dto.getCategoryId())) {
            FaqCategory newCategory = faqCategoryRepository.findById(dto.getCategoryId())
                    .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "FAQ Category"));
            faq.setCategory(newCategory);
        }

        return save(faq);
    }

    /**
     * Soft delete an FAQ by setting it to inactive.
     *
     * @param id The FAQ ID
     */
    @Transactional
    public void softDelete(Long id) {
        log.warn("GDPR: Service=softDelete, Operation=SOFT_DELETE, EntityID={}, Purpose=data_management", id);

        Faq faq = findById(id);
        faq.setActive(false);
        save(faq);

        log.info("GDPR: Service=softDelete_complete, EntityID={}, Status=inactive", id);
    }

    /**
     * Update FAQ display order.
     *
     * @param id       The FAQ ID
     * @param newOrder The new display order
     * @return The updated FAQ
     */
    @Transactional
    public Faq updateDisplayOrder(Long id, int newOrder) {
        Faq faq = findById(id);
        faq.setDisplayOrder(newOrder);
        return save(faq);
    }

    /**
     * Retrieves a paginated list of entities as DTOs with all conversions done within transaction.
     * This prevents LazyInitializationException by ensuring all DTO mappings happen inside @Transactional.
     *
     * @param pageable Pagination parameters
     * @param filters  Filter parameters
     * @param <DTOOUT> The output DTO type
     * @return Page of DTOs with all lazy relationships properly loaded
     */
    @Override
    public <DTOOUT> Page<DTOOUT> getDataPagedAndFilteredAsDtos(Pageable pageable, Map<String, String> filters) {
        // Get entities with proper pagination and filtering
        Page<Faq> page = getDataPagedAndFiltered(pageable, filters);

        // Convert to DTOs within transaction boundary
        Page<FaqDtoOut> dtoPage = page.map(entity -> {
            FaqDtoOut dto = new FaqDtoOut();

            // Map basic fields
            dto.setId(entity.getId());
            dto.setQuestion(entity.getQuestion());
            dto.setAnswer(entity.getAnswer());
            dto.setDisplayOrder(entity.getDisplayOrder());
            dto.setActive(entity.isActive());
            dto.setCreatedTime(entity.getCreatedTime());
            dto.setLastUpdateTime(entity.getLastUpdateTime());
            dto.setUpdaterId(entity.getUpdaterId());

            // Safely map category relationship
            if (entity.getCategory() != null) {
                // Force initialization of lazy-loaded category
                dto.setCategoryId(entity.getCategory().getId());
                dto.setCategoryName(entity.getCategory().getName());
            }

            return dto;
        });

        // Cast to generic type
        @SuppressWarnings("unchecked")
        Page<DTOOUT> result = (Page<DTOOUT>) dtoPage;
        return result;
    }

    /**
     * Convert entity to DTO within transaction.
     * Implements the abstract method from BaseService.
     *
     * @param entity The entity to convert
     * @return The converted DTO
     */
    @Override
    @Transactional(readOnly = true)
    public <DTOOUT> DTOOUT toDto(Faq entity) {
        if (entity == null) return null;
        @SuppressWarnings("unchecked")
        DTOOUT result = (DTOOUT) modelMapper.map(entity, FaqDtoOut.class);
        return result;
    }

    /**
     * Creates a new entity from DTO and returns it as DTO.
     * Implements the abstract method from BaseService.
     *
     * @param dto The input DTO
     * @return The created entity as DTO
     */
    @Override
    @Transactional
    public <DTOOUT> DTOOUT createFromDtoAsDto(FaqDtoIn dto) {
        @SuppressWarnings("unchecked")
        DTOOUT result = (DTOOUT) createFaqAsDto(dto);
        return result;
    }

    /**
     * Convert list of entities to DTOs within transaction.
     */
    @Transactional(readOnly = true)
    public List<FaqDtoOut> toDtoList(List<Faq> entities) {
        if (entities == null || entities.isEmpty()) return new ArrayList<>();
        return entities.stream()
                .map(entity -> (FaqDtoOut) toDto(entity))
                .collect(Collectors.toList());
    }

    /**
     * Get all active FAQs as DTOs.
     */
    @Transactional(readOnly = true)
    public List<FaqDtoOut> getAllActiveFaqsAsDto() {
        List<Faq> faqs = getAllActiveFaqs();
        return toDtoList(faqs);
    }

    /**
     * Get FAQs by category as DTOs.
     */
    @Transactional(readOnly = true)
    public List<FaqDtoOut> getFaqsByCategoryAsDto(Long categoryId) {
        List<Faq> faqs = getFaqsByCategory(categoryId);
        return toDtoList(faqs);
    }

    /**
     * Search FAQs and return as DTOs.
     */
    @Transactional(readOnly = true)
    public List<FaqDtoOut> searchFaqsAsDto(String query) {
        List<Faq> faqs = searchFaqs(query);
        return toDtoList(faqs);
    }

    /**
     * Create a new FAQ and return as DTO.
     */
    @Transactional
    public FaqDtoOut createFaqAsDto(FaqDtoIn dto) {
        Faq faq = createFaq(dto);
        return (FaqDtoOut) toDto(faq);
    }

    /**
     * Update an FAQ and return as DTO.
     */
    @Transactional
    public FaqDtoOut updateAsDto(Long id, FaqDtoIn dto) {
        Faq faq = update(id, dto);
        return (FaqDtoOut) toDto(faq);
    }

    /**
     * Find FAQ by ID and return as DTO.
     */
    @Transactional(readOnly = true)
    public FaqDtoOut findByIdAsDto(Long id) {
        Faq faq = findById(id);
        return (FaqDtoOut) toDto(faq);
    }

    /**
     * Update FAQ display order and return as DTO.
     */
    @Transactional
    public FaqDtoOut updateDisplayOrderAsDto(Long id, int newOrder) {
        Faq faq = updateDisplayOrder(id, newOrder);
        return (FaqDtoOut) toDto(faq);
    }
}