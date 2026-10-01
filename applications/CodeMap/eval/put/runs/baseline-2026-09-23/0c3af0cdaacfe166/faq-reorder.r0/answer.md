- `FaqReorderDtoIn.java` (new): request DTO with `faqIds` for the reorder endpoint.
- `FaqCategoryController.java`: added `PUT /support/faq/categories/{categoryId}/order` (admin-only), delegating to `FaqService.reorderCategory` and returning 204.
- `FaqService.java`: added `reorderCategory(Long, List<Long>)`, validating the ids match the category's active FAQs exactly (else `BusinessRuleTranslatableException` with key `error.business.faq_reorder_mismatch`), then assigning `displayOrder = i + 1` and saving each FAQ.
- `messages_en.properties` / `messages_pl.properties`: added `error.business.faq_reorder_mismatch` and `validation.faq.reorder.faq_ids.required` in both locales.
- `FaqServiceReorderUnitTest.java` (new): Mockito-only unit tests covering success reordering, missing/foreign/duplicate id mismatches, and unknown-category not-found behaviour.

Tests run: `FaqServiceReorderUnitTest` (pass), `FaqUnitTest` (pass), full `test-compile` (clean).