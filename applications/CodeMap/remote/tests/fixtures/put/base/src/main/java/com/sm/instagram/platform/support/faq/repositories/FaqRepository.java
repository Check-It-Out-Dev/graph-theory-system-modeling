package com.sm.instagram.platform.support.faq.repositories;

import com.sm.instagram.platform.common.base.BaseRepository;
import com.sm.instagram.platform.support.faq.models.Faq;
import com.sm.instagram.platform.support.faq.models.FaqCategory;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Repository for Faq entities.
 */
@Repository
public interface FaqRepository extends BaseRepository<Faq, Long> {

    /**
     * Find all active FAQs belonging to a category.
     *
     * @param category The category
     * @return List of active FAQs
     */
    List<Faq> findByCategoryAndActiveTrue(FaqCategory category);

    /**
     * Find all active FAQs belonging to a category, ordered by display order.
     *
     * @param category The category
     * @return List of active FAQs
     */
    List<Faq> findByCategoryAndActiveTrueOrderByDisplayOrderAsc(FaqCategory category);

    /**
     * Find all active FAQs ordered by display order.
     *
     * @return List of active FAQs
     */
    List<Faq> findByActiveTrueOrderByDisplayOrderAsc();

    /**
     * Search for FAQs containing the search term in question or answer.
     *
     * @param searchTerm The text to search for
     * @return List of matching FAQs
     */
    @Query("SELECT f FROM Faq f WHERE f.active = true AND (" +
            "LOWER(f.question) LIKE LOWER(CONCAT('%', :searchTerm, '%')) OR " +
            "LOWER(f.answer) LIKE LOWER(CONCAT('%', :searchTerm, '%')))")
    List<Faq> search(@Param("searchTerm") String searchTerm);

    /**
     * Get the maximum display order for a category.
     * Used when adding a new FAQ to a category.
     *
     * @param categoryId The category ID
     * @return The maximum display order, or 0 if no FAQs exist
     */
    @Query("SELECT COALESCE(MAX(f.displayOrder), 0) FROM Faq f WHERE f.category.id = :categoryId")
    int getMaxDisplayOrder(@Param("categoryId") Long categoryId);
}