Changes made:
- `src/main/java/.../support/faq/dtos/FaqReorderDtoIn.java` (new): `faqIds` list DTO.
- `src/main/java/.../support/faq/services/FaqService.java`: added `reorderCategory(Long, List<Long>)`, reusing `getFaqsByCategory` for not-found handling, validating the id set matches active FAQs exactly, then assigning `displayOrder = i + 1` and saving.
- `src/main/java/.../support/faq/FaqController.java`: added `PUT /support/faq/categories/{categoryId}/order` (admin-only, `@RateLimit` inherited from class), delegates to the service and returns 204.
- `messages_en.properties` / `messages_pl.properties`: added `error.business.faq_reorder_mismatch` and `validation.faq.reorder.faqIds.required` in both bundles.
- `src/test/java/.../unit/service/FaqServiceUnitTest.java` (new): covers unknown category, count/duplicate/foreign-id mismatches, single-FAQ and multi-FAQ valid reorders (asserting exact display orders and captured saved entities).

Tests run: `FaqServiceUnitTest`, `FaqUnitTest` — all passed; `test-compile -Ptest` green.