All checks pass: no LENIENT/lenient, `@ExtendWith(MockitoExtension.class)` present, changeset first line matches literal, has changeset line and rollback, and one new include was added to `changelog.xml`. No existing changeset was modified.

## Summary

- `src/main/java/com/sm/instagram/platform/userpreferences/UserPreferences.java`: added `@Version private Long version;` (JPA optimistic locking), following the same pattern as `PartnershipOpportunity`/`User`/`Notification`.
- `src/main/resources/db/changelog/2026/09/24-09-2026-add-version-to-user-preferences.sql`: new changeset adding `version BIGINT NOT NULL DEFAULT 0` to `user_preferences`, with rollback.
- `src/main/resources/db/changelog/changelog.xml`: added the new include.
- `src/test/java/com/sm/instagram/platform/userpreferences/UserPreferencesUnitTest.java`: new test asserting via reflection that `version` carries `@Version` and is `Long`, and that `UserPreferencesMapping` preserves an existing entity's version when a DTO is mapped onto it (and leaves it `null` for brand-new entities).
- Concurrent updates now fail fast with `ObjectOptimisticLockingFailureException`, already translated by the existing `BusinessExceptionHandler.handleOptimisticLocking`, so no controller/handler changes were needed.

Tests run: `UserPreferencesUnitTest` (new, pass), `UserPreferencesServiceUnitTest` and `UserPreferencesControllerUnitTest` (existing, pass), `test-compile -Ptest` (green).