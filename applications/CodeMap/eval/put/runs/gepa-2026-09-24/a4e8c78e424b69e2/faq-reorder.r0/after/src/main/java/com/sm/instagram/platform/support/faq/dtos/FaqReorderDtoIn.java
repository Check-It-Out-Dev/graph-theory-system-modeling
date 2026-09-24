package com.sm.instagram.platform.support.faq.dtos;

import jakarta.validation.constraints.NotEmpty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

/**
 * DTO for reordering the FAQs of a category in a single request.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class FaqReorderDtoIn {

    /**
     * The ids of the category's active FAQs, in their new display order.
     */
    @NotEmpty(message = "{validation.faq.reorder.faqIds.required}")
    private List<Long> faqIds;
}
