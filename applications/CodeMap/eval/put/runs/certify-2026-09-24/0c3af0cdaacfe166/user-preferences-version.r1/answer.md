All 169 tests pass. Build is green.

Added optimistic locking to `user_preferences` to prevent one device's edit from silently overwriting another's:

- `UserPreferences.java`: added `@Version private Long version;` field.
- `db/changelog/2026/09/24-09-2026-add-version-to-user-preferences.sql`: new changeset adding `version BIGINT NOT NULL DEFAULT 0` to `user_preferences`, with rollback.
- `changelog.xml`: included the new changeset.

No service/DTO changes were needed — a global `@ExceptionHandler(OptimisticLockingFailureException.class)` already exists (`BusinessExceptionHandler.java`) and returns HTTP 409 with the existing `error.database.concurrent_modification` message key, matching the same pattern used for `AppliedOpportunityContent`.

Tests run: `./mvnw -Ptest -Dtest=UserPreferencesServiceUnitTest,UserPreferencesControllerUnitTest -Dsurefire.failIfNoSpecifiedTests=false test` — 169 tests, 0 failures, BUILD SUCCESS.