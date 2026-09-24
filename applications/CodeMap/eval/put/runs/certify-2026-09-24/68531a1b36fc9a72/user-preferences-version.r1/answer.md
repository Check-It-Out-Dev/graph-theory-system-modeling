All checks pass: the changeset's first line is exactly `--liquibase formatted sql`, the entity field/column land together, the new test asserts `@Version` by reflection with `@ExtendWith(MockitoExtension.class)`, no `lenient`/`LENIENT`, and no `any()` on computed values.

## Summary

- `src/main/java/com/sm/instagram/platform/userpreferences/UserPreferences.java`: added `@Version @Column(name = "version") private Long version;` for optimistic locking.
- `src/main/java/com/sm/instagram/platform/userpreferences/UserPreferencesDtoOut.java`: added `version` field (ModelMapper auto-maps it by name), matching the `AppliedOpportunity`/`PartnershipOpportunity` precedent.
- `src/main/resources/db/changelog/2026/09/24-09-2026-add-version-to-user-preferences.sql`: new changeset adding `user_preferences.version BIGINT NOT NULL DEFAULT 0`, with rollback.
- `src/main/resources/db/changelog/changelog.xml`: new `<include>` with comment for the changeset above.
- `src/test/java/com/sm/instagram/platform/unit/service/UserPreferencesVersionUnitTest.java`: new test asserting the `@Version` annotation/type by reflection, and that mapping a `DtoIn` onto an existing entity (as the service does) preserves the version while mapping an entity to `DtoOut` carries it to the client.

Ran: `UserPreferencesVersionUnitTest`, `UserPreferencesServiceUnitTest`, `UserPreferencesControllerUnitTest` (all pass), and `test-compile -Ptest` (green, no errors).