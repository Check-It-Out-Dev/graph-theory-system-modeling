All checks pass: exact header literal, no lenient usage, `MockitoExtension` present, and only the needed files touched.

## Summary

- `src/main/java/com/sm/instagram/platform/userpreferences/UserPreferences.java`: added `@Version private Long version;` field for optimistic locking (following the `AppliedOpportunity` exemplar).
- `src/main/resources/db/changelog/2026/09/24-09-2026-add-version-to-user-preferences.sql`: new changeset adding `user_preferences.version BIGINT NOT NULL DEFAULT 0`, with rollback.
- `src/main/resources/db/changelog/changelog.xml`: included the new changeset.
- `src/test/java/com/sm/instagram/platform/unit/service/UserPreferencesServiceUnitTest.java`: added an `Optimistic Locking Tests` nested class — a reflection test asserting `version` is `Long` and carries `@Version`, and a test using the real `UserPreferencesMapping` configurer proving `modelMapper.map(dtoIn, preferences)` (as used in `updateCurrentUserPreferences`) preserves an existing version.

No service/mapping/DTO changes were needed: `UserPreferencesDtoIn` has no `version` property, so ModelMapper never touches it, and `OptimisticLockingFailureException` is already translated by the global `BusinessExceptionHandler`.

Tests run: `UserPreferencesServiceUnitTest` (pass), `UserPreferencesControllerUnitTest` (pass), `mvnw test-compile -Ptest` (green).