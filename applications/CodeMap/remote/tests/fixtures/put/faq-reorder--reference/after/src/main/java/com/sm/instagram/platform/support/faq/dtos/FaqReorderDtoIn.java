package com.sm.instagram.platform.support.faq.dtos;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

/**
 * The new order of a category's active FAQs: every id once, first shown first.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class FaqReorderDtoIn {

    @NotEmpty
    private List<@NotNull Long> faqIds;
}
