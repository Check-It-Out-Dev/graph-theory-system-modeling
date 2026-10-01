All changes are scoped to the FAQ reorder feature.

Summary:
- `FaqReorderDtoIn.java` (new): request DTO with `faqIds`, following the `FaqDtoIn` exemplar.
- `FaqService.java`: added `reorderCategory(Long, List<Long>)` validating the id set matches the category's active FAQs exactly, assigning display order `i+1`, saving each; throws `BusinessRuleTranslatableException("error.business.faq_reorder_mismatch")` on mismatch, reuses existing not-found path for unknown categories.
- `FaqCategoryController.java`: added `PUT /support/faq/categories/{categoryId}/order` (`@PreAuthorize("hasAuthority('ADMIN')")`, class-level `@RateLimit` applies), returns 204; injected `FaqService`.
- `messages_en.properties` / `messages_pl.properties`: added `error.business.faq_reorder_mismatch` key in both bundles.
- `FaqServiceUnitTest.java` (new): covers success reordering (captures actual assigned orders), empty category boundary, missing/duplicate/foreign id mismatches (asserting exception type, message key, and no save), and unknown category.

Tests run: `mvn test -Ptest -Dtest=FaqUnitTest,FaqServiceUnitTest` — passed; `mvn test-compile -Ptest` — green.