Checks pass: both new test classes carry `@ExtendWith(MockitoExtension.class)`, no `LENIENT`/`lenient(`, no adapter imports (n/a, no ports/adapters involved). Message keys, `@PreAuthorize`, and class-level `@RateLimit` were verified during implementation.

## Summary

- `src/main/java/.../support/faq/dtos/FaqReorderDtoIn.java` (new): `FaqReorderDtoIn` with `faqIds` (`List<Long>`).
- `src/main/java/.../support/faq/services/FaqService.java`: added `reorderCategory(Long, List<Long>)` — validates the id set matches the category's active FAQs exactly (no dupes/missing/foreign ids), sets display order `i+1` and saves each FAQ; reuses the existing not-found path for an unknown category; throws `BusinessRuleTranslatableException("error.business.faq_reorder_mismatch")` on mismatch.
- `src/main/java/.../support/faq/FaqController.java`: added `PUT /support/faq/categories/{categoryId}/order`, `@PreAuthorize("hasAuthority('ADMIN')")`, delegates to the service, returns 204.
- `src/main/resources/messages_en.properties`, `messages_pl.properties`: added `error.business.faq_reorder_mismatch` in both.
- New tests: `src/test/java/.../support/faq/services/FaqServiceUnitTest.java` (main-path ordering + not-found + mismatch cases: missing, duplicate, foreign, null ids) and `src/test/java/.../support/faq/FaqControllerUnitTest.java` (delegation + 204 response).
- Ran: `FaqServiceUnitTest`, `FaqControllerUnitTest`, `FaqUnitTest` — all passed; `test-compile -Ptest` — green.