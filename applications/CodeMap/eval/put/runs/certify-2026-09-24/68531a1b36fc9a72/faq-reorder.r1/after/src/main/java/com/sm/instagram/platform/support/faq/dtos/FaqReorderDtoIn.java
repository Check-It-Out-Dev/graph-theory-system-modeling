package com.sm.instagram.platform.support.faq.dtos;

import lombok.AllArgsConstructor;
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
@AllArgsConstructor
public class FaqReorderDtoIn {

    /**
     * The ids of the category's active FAQs, in the desired display order.
     */
    private List<Long> faqIds;
}
