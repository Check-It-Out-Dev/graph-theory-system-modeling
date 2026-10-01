Endpoint is guarded by `@PreAuthorize` and the class-level `@RateLimit`, matches `dto_naming` (`FaqReorderDtoIn` in, no body out), and returns 204. All changes are scoped to the FAQ reorder feature.

## Summary

- `FaqService.java`: added `reorderCategory(Long categoryId, List<Long> faqIds)` — validates the given ids are exactly the category's active FAQ ids (each once), assigns display order `i+1`, saves each FAQ; throws `BusinessRuleTranslatableException("error.business.faq_reorder_mismatch")` on mismatch, keeps existing not-found behaviour for unknown categories.
- `FaqCategoryController.java`: added `PUT /support/faq/categories/{categoryId}/order` (`reorderFaqs`), `@PreAuthorize("hasAuthority('ADMIN')")`, delegates to `FaqService`, returns 204; constructor now also injects `FaqService`.
- `dtos/FaqReorderDtoIn.java` (new): `List<Long> faqIds` with getter/setter.
- `messages_en.properties` / `messages_pl.properties`: added `error.business.faq_reorder_mismatch` key in both bundles.
- `test/.../support/faq/services/FaqServiceUnitTest.java` (new): covers reorder success (order assignment + saves), size mismatch, duplicate id, foreign id, unknown category.
- `test/.../support/faq/FaqCategoryControllerUnitTest.java` (new): verifies delegation and 204 response.

Tests run: `FaqServiceUnitTest`, `FaqCategoryControllerUnitTest`, `FaqUnitTest` (existing) — all passed; `mvnw test-compile -Ptest` green.