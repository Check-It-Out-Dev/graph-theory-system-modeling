All 7 tests pass (grew from 6→7). Changes are complete and verified.

## Summary

- `src/main/java/com/sm/instagram/platform/userpreferences/UserPreferences.java`: added `@Version private Long version;` for JPA optimistic locking.
- `src/main/resources/db/changelog/2026/09/24-09-2026-add-version-to-user-preferences.sql`: new Liquibase changeset adding `version BIGINT NOT NULL DEFAULT 0` to `user_preferences`, with rollback.
- `src/main/resources/db/changelog/changelog.xml`: included the new changeset.
- No service/controller/handler changes needed — `BusinessExceptionHandler.handleOptimisticLocking` already translates `OptimisticLockingFailureException` to `error.database.concurrent_modification` (present in both `messages_en.properties` and `messages_pl.properties`), so a losing concurrent write now returns HTTP 409 instead of silently overwriting.
- `src/test/java/com/sm/instagram/platform/unit/service/UserPreferencesServiceUnitTest.java`: added a version getter/setter test and a test asserting a stale `save()` surfaces `ObjectOptimisticLockingFailureException`.

Tests run: `UserPreferencesServiceUnitTest` (all nested classes, 60 tests incl. 2 new) and `UserPreferencesControllerUnitTest` (all nested classes) — all passed, 0 failures/errors. Full `mvn compile -Ptest` also succeeded.