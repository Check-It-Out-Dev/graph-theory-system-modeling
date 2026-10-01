Working tree matches the intended scope, nothing left uncommitted beyond the task's needs.

- `FaqReorderDtoIn.java` (new): `faqIds` list DTO for the reorder request.
- `FaqCategoryController.java`: added `PUT /support/faq/categories/{categoryId}/order` (`@PreAuthorize("hasAuthority('ADMIN')")`, `@RateLimit`), returns 204; injected `FaqService`.
- `FaqService.java`: added `reorderCategory(Long, List<Long>)`, reusing `getFaqsByCategory` for the not-found/active-FAQ lookup, validating the set/size match and throwing `BusinessRuleTranslatableException("error.business.faq_reorder_mismatch")` on mismatch.
- `messages_en.properties` / `messages_pl.properties`: added `error.business.faq_reorder_mismatch` and `validation.faq.reorder.faqIds.required` in both bundles.
- New test `FaqServiceUnitTest.java` (Mockito, no Spring context): covers happy-path reordering, missing FAQ, duplicate id, foreign id, and unknown category.

Ran `./mvnw -q test -Ptest -Dtest=FaqServiceUnitTest,FaqUnitTest -Dsurefire.failIfNoSpecifiedTests=false` — all 5 new tests and all existing `FaqUnitTest` tests pass. Full `mvn compile` also succeeded.