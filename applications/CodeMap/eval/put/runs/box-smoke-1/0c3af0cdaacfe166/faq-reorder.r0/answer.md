Compilation is clean. Summary of changes:

- `src/main/java/.../support/faq/dtos/FaqReorderDtoIn.java` (new): DTO with `faqIds` list.
- `src/main/java/.../support/faq/services/FaqService.java`: added `reorderCategory(Long categoryId, List<Long> faqIds)`, validating the ids exactly match the category's active FAQs and assigning display order `i+1`, throwing `BusinessRuleTranslatableException("error.business.faq_reorder_mismatch")` on mismatch.
- `src/main/java/.../support/faq/FaqController.java`: added `PUT /support/faq/categories/{categoryId}/order`, `@PreAuthorize("hasAuthority('ADMIN')")`, returns 204.
- `messages_en.properties` / `messages_pl.properties`: added `error.business.faq_reorder_mismatch` key in both.
- `src/test/java/.../unit/service/FaqServiceUnitTest.java` (new): covers success reorder, unknown category, missing id, unknown id, and duplicate id cases.

Tests run: `FaqServiceUnitTest` (new, 5 tests pass) and `FaqUnitTest` (existing, unaffected) via `./mvnw -q test -Ptest -Dtest=... -Dsurefire.failIfNoSpecifiedTests=false`; full `mvn compile -Ptest` also succeeds.