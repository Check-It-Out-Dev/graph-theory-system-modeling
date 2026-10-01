Clean. Everything builds and the new tests pass.

## Summary

- `src/main/java/.../support/faq/dtos/FaqReorderDtoIn.java` (new): `List<Long> faqIds` request DTO.
- `src/main/java/.../support/faq/services/FaqService.java`: added `reorderCategory(Long categoryId, List<Long> faqIds)` — validates the submitted ids are exactly the category's active FAQ ids, each once (else `BusinessRuleTranslatableException` with key `error.business.faq_reorder_mismatch`), then sets `displayOrder = i + 1` and saves each FAQ. Unknown category keeps existing `ResourceNotFoundException` behaviour.
- `src/main/java/.../support/faq/FaqCategoryController.java`: injected `FaqService`; added `PUT /support/faq/categories/{categoryId}/order` (`@PreAuthorize("hasAuthority('ADMIN')")`), returns 204.
- `messages_en.properties` / `messages_pl.properties`: added `error.business.faq_reorder_mismatch` and `validation.faq.faqIds.required` in both locales.
- `src/test/java/.../support/faq/services/FaqServiceUnitTest.java` (new): 7 unit tests covering reordering, order-agnostic acceptance, mismatch (missing/unknown/duplicate/null ids), and not-found category.

Tests run: `./mvnw -q test -Ptest -Dtest=FaqServiceUnitTest -Dsurefire.failIfNoSpecifiedTests=false` — 7/7 passed. Also ran `mvnw compile` and `test-compile` with `-Ptest` — both clean.