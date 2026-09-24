Everything is scoped to the task. Summary:

- `FaqReorderDtoIn.java` (new): `List<Long> faqIds` request DTO.
- `FaqService.java`: added `reorderCategory(Long categoryId, List<Long> faqIds)` — validates the ids match exactly the category's active FAQs (each once), sets display order `i+1` per position and saves each, else throws `BusinessRuleTranslatableException("error.business.faq_reorder_mismatch")`; unknown category keeps the existing `ResourceNotFoundException`.
- `FaqCategoryController.java`: injected `FaqService`, added `PUT /support/faq/categories/{categoryId}/order` (`@PreAuthorize("hasAuthority('ADMIN')")`, class-level `@RateLimit` already covers it), returns 204.
- `messages_en.properties` / `messages_pl.properties`: added the new key in both bundles.
- `FaqServiceUnitTest.java` (new): covers reorder success (order assignment + save calls captured), missing id, duplicate id, id from another category, and unknown category.

Tests run: `FaqServiceUnitTest`, `FaqUnitTest` (pass); `test-compile -Ptest` green.