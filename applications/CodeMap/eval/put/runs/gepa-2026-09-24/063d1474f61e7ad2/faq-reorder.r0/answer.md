All changes are scoped to the FAQ reorder feature as required.

## Summary

- `FaqReorderDtoIn.java` (new): request DTO holding `List<Long> faqIds`.
- `FaqService.java`: added `reorderCategory(Long categoryId, List<Long> faqIds)` — validates the ids match the category's active FAQs exactly (each once), assigns display order `i+1`, saves each; unknown category still throws `ResourceNotFoundException`; mismatch throws `BusinessRuleTranslatableException("error.business.faq_reorder_mismatch")`.
- `FaqCategoryController.java`: added `PUT /support/faq/categories/{categoryId}/order` (`@PreAuthorize("hasAuthority('ADMIN')")`, covered by class-level `@RateLimit`), returns 204; injected `FaqService` via constructor.
- `messages_en.properties` / `messages_pl.properties`: added `error.business.faq_reorder_mismatch` and `validation.faq.faqIds.required` in both bundles.
- `FaqServiceUnitTest.java` (new): Mockito-only unit tests covering the reordered-save path (captures and asserts saved display orders), missing-id and duplicate-id mismatch paths, and the not-found path.

Tests run: `FaqServiceUnitTest` (4/4 passed), `test-compile -Ptest` (clean, whole project compiles).