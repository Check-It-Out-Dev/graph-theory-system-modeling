All checks pass: changeset literal matches, no LENIENT/lenient, `@ExtendWith(MockitoExtension.class)` present, test package matches its directory.

## Summary

- `UserPreferences.java`: added `@Version private Long version;` field for optimistic locking.
- `db/changelog/2026/09/24-09-2026-add-version-to-user-preferences.sql`: new changeset adding `version BIGINT DEFAULT 0 NOT NULL` to `user_preferences`, with rollback.
- `changelog.xml`: new include for the changeset above.
- `UserPreferencesUnitTest.java` (new, package `com.sm.instagram.platform.userpreferences`): asserts the field carries `@Version` and is `Long`, and that mapping a `UserPreferencesDtoIn` onto an existing entity preserves its loaded version.

Tests run: `UserPreferencesUnitTest`, `UserPreferencesServiceUnitTest`, `UserPreferencesControllerUnitTest` — 171 tests, all passing. `mvn test-compile -Ptest` — green.