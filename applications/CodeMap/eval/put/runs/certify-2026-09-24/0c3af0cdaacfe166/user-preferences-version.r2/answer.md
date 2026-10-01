Changes complete, working tree left uncommitted as instructed.

- `src/main/java/com/sm/instagram/platform/userpreferences/UserPreferences.java`: added `@Version private Long version;` (`version` column), following the `AppliedOpportunity` optimistic-locking exemplar — JPA now rejects a stale save with `OptimisticLockingFailureException` (already handled globally by `BusinessExceptionHandler` → `error.database.concurrent_modification`, already translated in both `messages_en/pl.properties`), so a device saving over another's concurrent commit gets a 409 instead of silently overwriting it.
- `src/main/resources/db/changelog/2026/09/24-09-2026-add-version-to-user-preferences.sql` (new): Liquibase changeset adding `version BIGINT DEFAULT 0 NOT NULL` to `user_preferences`, with rollback.
- `src/main/resources/db/changelog/changelog.xml`: included the new changeset.
- `src/test/java/com/sm/instagram/platform/unit/service/UserPreferencesServiceUnitTest.java`: added two tests to `UserPreferencesEntityTests` covering the new `version` field.

I deliberately did not touch `UserPreferencesDtoIn`/`DtoOut`/service mapping: since those DTOs have no `version` property, ModelMapper's default matching leaves the freshly-loaded entity's version untouched, which is what keeps `JpaRepository.save()`'s `isNew()` check (based on the version property being non-null) working correctly — wiring a client-supplied `version` through those DTOs would risk nulling it out on older payloads and misclassifying updates as inserts.

Tests run: `UserPreferencesServiceUnitTest` (all nested classes, including the 2 new tests) and `UserPreferencesControllerUnitTest` — all passed, no failures. Full `mvn compile` also succeeded.