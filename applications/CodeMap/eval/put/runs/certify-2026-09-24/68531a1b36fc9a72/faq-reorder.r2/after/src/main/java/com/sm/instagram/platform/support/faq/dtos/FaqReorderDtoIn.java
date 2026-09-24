package com.sm.instagram.platform.support.faq.dtos;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

/**
 * DTO for reordering the FAQs of a category.
 */
@Getter
@Setter
@NoArgsConstructor
public class FaqReorderDtoIn {

    /**
     * Ids of the category's FAQs, in the desired display order.
     */
    private List<Long> faqIds;
}
