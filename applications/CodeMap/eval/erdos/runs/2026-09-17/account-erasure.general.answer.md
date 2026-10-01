## Problem

An influencer wants their account erased. The code has five ways into deletion, and none of them erases everything. The usual path from the app never actually erases anything. The hard-delete paths either fail on database constraints, leave files and cache data behind, or destroy consent records the law says we must keep. We need one erasure design that removes personal data from every store and keeps only records the law requires, with personal details stripped out.

## Where it lives today

**Entry points (backend, `backend/src/main/java/com/sm/instagram/platform/…`)**

| # | Trigger | Code | What it does |
|---|---|---|---|
| 1 | Influencer clicks "delete account" in the app | FE `feature/profile/profile-view.component.ts:208` → `DELETE /users/{id}` → `user/UserService.java:342` → `UserAccountOrchestrator.archiveUser` | Sets status to `TO_BE_DELETED` and bumps the token version. Nothing runs after that. |
| 2 | Meta data-deletion callback | `auth/controller/InstagramCallbackController.java:59` → `auth/service/InstagramDataDeletionService.java` | No blockers: calls `archiveUser` and stores a COMPLETED request. Blockers: sets `TO_BE_DELETED` and stores a PENDING `PendingDataDeletionRequest`. |
| 3 | Deferred deletion | `auth/cron/DeferredDeletionCronJob.java` | Rechecks only PENDING requests and calls `archiveUser` once the influencer is eligible. |
| 4 | Meta deauthorization callback | `auth/service/InstagramDeauthorizationService.java` | Verified email: disconnects Instagram, deletes the Firestore token doc, sends an email. Unverified email: `archiveUser`. |
| 5 | Admin hard delete | `admin/cascade/AdminCascadeDeleteServiceImpl.forceDeleteUser`, `UserService.deletePermanently`, `auth/service/FirebaseAuthProxyService.deleteAccount` (`DELETE /auth/delete-account`) | Deletes the database row, plus Firestore, Storage and Firebase Auth in the cascade version only. |
| 6 | No-consent cleanup | `legal/LegalConsentService.archiveUsersWithNoConsents:540` | `archiveUser` |

**Eligibility and blockers:** `UserAccountOrchestrator.checkSoftDeleteBlockers` blocks an influencer who has active or APPLIED applications. The "permanent delete" and support-ticket blockers are empty placeholders (`:221`, `:228`). The frontend comment says the blocker check is client-side only (`core/user/user.service.ts:71`). In practice `archiveUser` still throws on active applications and silently deletes APPLIED ones.

**Stores that hold influencer data**
- **Database:** `user` (email, names, phone, profile picture, NIP, last verified email, admin note), `address`, `user_social_connection`, `user_preferences`, `notifications` (plus snapshots of the influencer shown in *other* users' notifications, `notification/ActorSnapshot.java`), `applied_opportunity` / `_content` / `_status_history` (`changed_by_firebase_id`), `support_ticket` (`contact_email`, `ip_address`), `search_query_log.ip_address`, `consent_record` (IP, user agent, proof JSON), `file_uploads` (keyed by Firebase UID, no foreign key), `pending_data_deletion_request`.
- **Firestore:** `instagramUsers/{uid}` and the TOTP secret data.
- **Firebase Storage:** `users/{uid}/`, `content/{uid}/` (the default upload path, `storage/service/SignedUrlService.java:65`) and `profile-pictures/{uid}/` (`ProfilePictureProxyService.java:95`).
- **Redis:** `user_cache` (5-minute TTL), rate-limit audit data (`GdprCompliantRateLimiterService.deleteUserRateLimitData`), geo travel patterns (`GeoLocationGdprService.deleteUserLocationData`), storage quota.
- **Firebase Auth:** the login account itself.

### Where personal data survives or the flow breaks
1. **Deleting from the app never erases anything.** `archiveUser` for an influencer only clears two email-verification timestamps (`UserAccountOrchestrator.java:262`). Name, email, phone, profile picture, addresses, social connections, Firestore tokens, files and the Firebase Auth account all stay. The cron only handles `PendingDataDeletionRequest` rows, and only the Meta callback creates those. The frontend comment "DeferredDeletionCronJob performs the RODO cascade later" is wrong. Nothing ever moves an account from `TO_BE_DELETED` to `DELETED`: no code sets `AccountStatus.DELETED` or `deletedAt`.
2. **Influencers and companies are anonymized differently.** Companies get their names and phone replaced (`:275`); influencers don't. The tests only check the company case (`UserAccountOrchestrator_Archive_IntegrationTest.java:338`).
3. **Admin cascade delete misses files.** It deletes only `users/{uid}/` (`FirebaseStorageService.java:48`). `content/{uid}/`, `profile-pictures/{uid}/`, `file_uploads` rows, Redis rate-limit and geo data, notification snapshots, support-ticket email/IP and status-history UIDs all stay.
4. **Admin cascade delete touches external systems before the database commits.** Firebase Auth, Storage and Firestore are deleted inside the `@Transactional` method (`:300-347`). If the commit then fails, the Firebase account is gone but the database row stays, and the task row that recorded the attempt is rolled back with it.
5. **A failed Firestore delete is reported as SKIPPED, not FAILED.** `deleteInstagramUserData` returns `false` on error, and the cascade treats `false` as SKIPPED (`:304`). The task then counts as complete and is never retried.
6. **Hard deletes break on foreign keys.** `pending_data_deletion_request.user_id` has no ON DELETE rule, and `applied_opportunity.influencer_id` is `NOT NULL … ON DELETE SET NULL`, which contradicts itself. So `deletePermanently` and `/auth/delete-account` can't hard-delete an influencer who has any application or any Meta deletion request.
7. **Consent proof is destroyed.** `consent_record.user_id` is `ON DELETE SET NULL`, and the weekly cleanup deletes `user IS NULL AND timestamp < now-1y` (`ConsentRecordRepository.java:69`). A hard delete turns the user's consent history into "anonymous" rows, which the next Sunday run removes if they are over a year old. Until then the IP address, user agent and screen coordinates stay stored, just unlinked.
8. **Meta's status page doesn't exist.** The frontend has no `/deletion-status` route; the catch-all redirects to 404 (`app.routes.ts:619`). If processing throws, the controller still answers 200 with a made-up code and saves no record (`InstagramCallbackController.java:66-73`), so a failed deletion can't be traced.
9. **Deferred deletion can deadlock.** The deferred path locks the influencer out with `TO_BE_DELETED` while the collaborations that block deletion may need the influencer's own action (see VERIFIED).
10. **Deauthorization is incomplete.** A verified user keeps a `DISCONNECTED` connection row with display name, email, profile URL and follower count. An unverified user's archival failure is swallowed (`:114`).

## Proposed change

**One erasure pipeline. The user row becomes a tombstone; everything else is erased or anonymized, and each step is recorded in a ledger.**

1. **`ErasureRequest` (extends `pending_data_deletion_request`).** Every trigger (app, Meta, admin, no-consent cron) creates one. It has `source`, `status` (`REQUESTED → BLOCKED → GRACE → ERASING → ERASED | FAILED`), `confirmation_code`, a pseudonymous `subject_hash`, and one status per system, reusing the `CascadeDeleteTask`/`SystemStatus` model. `user_id` becomes nullable with `ON DELETE SET NULL`, so the request itself survives as proof we did the erasure (GDPR Art. 5(2)).
2. **Blocker gate on the server.** If blockers exist, the request goes to BLOCKED. The influencer gets a restricted "pending deletion" login limited to finishing collaborations, or admins/companies are notified to close them. This is a product decision; the current full lockout deadlocks.
3. **Grace window.** Optional, e.g. 14 days, during which the account can be reactivated (matches `TO_BE_DELETED → ACTIVE`). After it, erasure can't be undone.
4. **`ErasureExecutor`: idempotent, runs step by step, retried by cron.**
   - **Database, in one transaction:**
     - Overwrite the user row: email, names, `name`, phone, profile picture, NIP, admin note and `lastVerifiedEmail` become null; `firebase_user_id` becomes `erased-{id}`; status becomes `DELETED`; `deletedAt` is set.
     - Delete addresses, social connections, preferences, the user's own notifications and APPLIED applications.
     - In retained applications, clear content URLs, descriptions and tags.
     - Null status-history `changed_by_firebase_id`.
     - Rewrite other users' notification snapshots where `influencer_id` matches to `name="Deleted user"` and `avatarUrl=null`.
     - Redact support-ticket `contact_email` and `ip_address`, and null `search_query_log.ip_address`.
     - Delete `file_uploads` rows for the UID.
     - Consent records keep document, version, hash, timestamp and source. Truncate the IP, drop the user agent and screen coordinates, and stay linked to the tombstone so the anonymous-cleanup cron can't match them.
   - **After commit (outbox pattern):**
     - Firestore: `instagramUsers` and TOTP.
     - Storage prefixes `users/`, `content/` and `profile-pictures/{uid}/`.
     - Redis: user cache, rate-limit data, geo data and storage quota.
     - Firebase Auth: last.
     - Every failure is recorded as FAILED and retried.
5. **Keep the tombstone instead of hard-deleting the row.** Records the company or the law needs keep a valid foreign key and hold no personal data:
   - DONE/TO_BE_PAID collaborations: company business records.
   - Consent proof: Art. 7(1).
   - The erasure ledger.

   This also avoids the foreign-key failures in point 6. Admin "force delete" becomes the same executor with an extra option to delete collaborations, instead of a separate half-finished path.
6. **Retention job.** Deletes consent records and tombstones once their legal retention period ends. The period is a legal decision; about 6 years for claims is my assumption, not something in the code.

## Plan

1. **Migration.** `db/changelog/2026/09/…-erasure-pipeline.sql`:
   - Alter `pending_data_deletion_request`: nullable `user_id` with `ON DELETE SET NULL`; add `source`, `subject_hash`, the per-system status columns and `grace_until`.
   - Fix `applied_opportunity.influencer_id` so it is either `RESTRICT` or nullable.
   - Add the new statuses.
   - Wire it into `changelog.xml`.
2. **Entities and repositories.** `auth/entity/PendingDataDeletionRequest.java`, `DeletionRequestStatus.java`, `PendingDataDeletionRequestRepository.java`. Add `findDue…` and `findBySubjectHash`.
3. **`user/erasure/ErasureService`** (create request, apply blockers, grace window) **and `ErasureExecutor`** (database tombstone and redaction). Add redaction queries to `NotificationRepository`, the support-ticket repositories, `AppliedOpportunityStatusHistoryRepository`, `AppliedOpportunityContentRepository`, `ConsentRecordRepository` and `FileUploadRepository`.
4. **External cleanup** in `ErasureExecutor`, after commit:
   - Add `FirebaseStorageService.deleteUserData(uid)`, which deletes all three prefixes.
   - Make `FirestoreService.deleteInstagramUserData` throw on error instead of returning `false`.
   - Call `GdprCompliantRateLimiterService.deleteUserRateLimitData`, `GeoLocationGdprService.deleteUserLocationData` and the storage quota reset.
5. **Point every trigger at `ErasureService`:**
   - `UserService.delete` and `deletePermanently`
   - `FirebaseAuthProxyService.deleteAccount`: remove it or delegate
   - `InstagramDataDeletionService`, `InstagramDeauthorizationService`
   - `LegalConsentService.archiveUsersWithNoConsents`
   - `AdminCascadeDeleteServiceImpl.forceDeleteUser` / `retryTask`
6. **Replace `DeferredDeletionCronJob`** with an `ErasureCronJob`: re-evaluate BLOCKED requests, expire GRACE, retry FAILED steps.
7. **Make `archiveUser`** (`UserAccountOrchestrator.java`) only lock the account; all anonymization moves into the executor.
8. **Consent retention.** In `LegalConsentService.cleanupAnonymousRecords`, make the delete target real anonymous cookie consents only (e.g. `source = COOKIE_BANNER`). Add a retention job for records tied to tombstones.
9. **`InstagramCallbackController`.** On an exception, save a FAILED request under the code it returns, so the failure can be traced.
10. **Frontend:**
    - Add a `/deletion-status` route and page (`app.routes.ts`, new `feature/deletion-status/`) using `instagram-callback-controller.api.ts`.
    - `profile-view.component.ts` and `user.service.ts`: show the confirmation code, deferred state and grace window; fix the misleading comment.
    - Admin view of the erasure ledger (`core/admin/cascade-delete.service.ts`).
    - Regenerate the API client.
11. **Tests** (next section).

## Risks and invariants

| Invariant | Guarding test |
|---|---|
| After ERASED, no column in `user`, `address`, `user_social_connection`, `notifications` (snapshots included), `support_ticket`, `status_history`, `file_uploads` or `consent_record` contains the influencer's email, names, phone, UID or IP | New `ErasureExecutor_PiiSweep_IntegrationTest`: seed marker strings everywhere, run erasure, search every text/jsonb column for the markers |
| Influencer and company archival both anonymize | Extend `UserAccountOrchestrator_Archive_IntegrationTest` with an influencer case |
| Firebase Auth, Storage and Firestore are touched only after the database commits; a failure leaves a FAILED step that gets retried | `ErasureExecutorUnitTest` (commit fails → no external calls); `ErasureCronJobUnitTest` (retries FAILED) |
| All 3 storage prefixes are deleted | `FirebaseStorageServiceUnitTest` |
| Consent records for erased users survive the anonymous cleanup | `AnonymousConsentCleanupCronJobUnitTest` plus an integration test |
| DONE/TO_BE_PAID collaborations stay for the company, with the influencer shown as "Deleted user" | `ErasureExecutor_RetainedRecords_IntegrationTest` |
| Callbacks stay idempotent and always return 200; the confirmation code resolves on the status page | Extend `InstagramDataDeletionServiceUnitTest`; FE spec for the status page; e2e `consent-lifecycle` / `admin-user-management` |
| A BLOCKED request doesn't deadlock | Integration test: blocked request, company closes the collaboration, cron erases |
| Last admin can't be erased | Existing blocker tests |

Other risks:
- **Backups** still hold data. An erasure ledger that is replayed after any restore is needed; this is a process, not code.
- **Logs:** plaintext emails appear in logs (e.g. `LegalConsentService.java:547`); log retention has to be bounded.
- **Firebase Auth deleted but tombstone kept:** re-registering with the same email creates a new user, which is fine because the email was nulled.

## Evidence

- FACT: influencer `archiveUser` only clears verification timestamps; the company branch anonymizes names and phone (`UserAccountOrchestrator.java:244-281`).
- FACT: the cron only processes PENDING requests (`DeferredDeletionCronJob.java:57`); requests are built only in `InstagramDataDeletionService.java:139,170`.
- FACT: no code sets `AccountStatus.DELETED` or `setDeletedAt` (grep over `src/main/java`).
- FACT: the frontend deletes via `DELETE /users/{id}` and its comment claims the cron cascades later (`user.service.ts:78-85`, `profile-view.component.ts:246`).
- FACT: cascade Storage cleanup covers only `users/{uid}/` (`FirebaseStorageService.java:48`); uploads default to `content/{userId}/` (`SignedUrlService.java:65`); profile pictures go to `profile-pictures/{uid}/` (`ProfilePictureProxyService.java:95`).
- FACT: `file_uploads.userId` is a plain string with no foreign key (`FileUpload.java:31`).
- FACT: cascade external cleanup runs inside the `@Transactional` `forceDeleteUser` (`AdminCascadeDeleteServiceImpl.java:243-376`).
- FACT: `deleteInstagramUserData` returns `false` on error (`FirestoreService.java:434-444`), and the cascade maps `false` to SKIPPED (`:304`). `isFullyCompleted` counts SKIPPED as done (`CascadeDeleteTask.java:190`).
- FACT: foreign keys: `pending_data_deletion_request` has no ON DELETE rule (`11-03-2026-create-pending-data-deletion-request.sql:17`); `applied_opportunity.influencer_id NOT NULL … ON DELETE SET NULL` (`002-tables.sql:187`); `consent_record.user_id ON DELETE SET NULL` (`09-03-2026-consent-module-tables.sql:36`). No later migration alters them (grep of `2025/` and `2026/`). `ddl-auto: validate` (`application.yml:45`).
- FACT: anonymous consent cleanup deletes `user IS NULL AND timestamp < now-1y` (`ConsentRecordRepository.java:69`, `LegalConsentService.java:571`).
- FACT: `User` cascades only to addresses and social connections (`User.java:68,107`).
- FACT: the Redis and geo erasure methods are called only from their own privacy controllers (grep).
- FACT: there is no frontend `deletion-status` route (grep of `frontend/src/app` finds it only in generated API files); the catch-all goes to 404 (`app.routes.ts:619`).
- FACT: the callback returns a made-up code on an exception without saving anything (`InstagramCallbackController.java:66-73`).
- FACT: notification snapshots store the actor's name and avatar (`ActorSnapshot.java:60-75`); `support_ticket` has `contact_email` and `ip_address` (`002-tables.sql:275,281`).
- INFERENCE: hard delete of an influencer with applications or Meta requests fails in Postgres. This follows from the foreign keys plus the lack of JPA cascade.
- INFERENCE: the cascade's database deletes run at commit, after the Firebase calls, so an FK failure leaves Firebase deleted, the database row intact and no task row.
- HYPOTHESIS: the deferred path deadlocks because blocking statuses need the influencer to act while `TO_BE_DELETED` can't log in.
- HYPOTHESIS: the legal retention periods (about 6 years for claims; accounting periods for invoices) need legal counsel. They are not in the code.

## Corrections after verification

I re-checked the claims below by reading more code. I didn't run anything, so the database and transaction-timing points are still reasoned from the code rather than observed.

1. **The deferred-deletion deadlock (was HYPOTHESIS) is confirmed from the code.**
   - `JwtAuthenticationFilter.java:196` returns 401 for any user who isn't active, and `TO_BE_DELETED` counts as not active (`RedisUserCache.java:50`).
   - The application states that count as blockers are listed in `AppliedOpportunityService.java:1176-1185`.
   - `OpportunityStatus.getNextStatus` / `canTransitionTo` (`OpportunityStatus.java:134-212`) have no cancel option in any of them.
   - Five of the seven blocking states only move forward through the influencer's own action: `ACCEPTED_BY_COMPANY`, `ACCEPTED_BY_INFLUENCER`, `CONTENT_REJECTED`, `CONTENT_APPROVED`, `CONTENT_POSTED_REJECTED`. Only `CONTENT_SEND_TO_ACCEPT` and `CONTENT_POSTED` can be moved by the company.
   - So a Meta request that gets deferred stays PENDING forever unless an admin steps in. I didn't check whether admins can override a status; if they can't, `forceDeleteUser` is the only way out.
   - This makes the BLOCKED handling in design point 2 required, not optional.

2. **Hard delete failing for an influencer with applications or a Meta request (INFERENCE) still holds, but no test covers it.**
   - No XML/YAML changelog changes those foreign keys (grep of `src/main/resources`), and `User` has no mapping to applications or deletion requests.
   - `UserService_Delete_IntegrationTest.adminPermanentlyDeletesUser` (`:112`) uses an influencer with no applications, so it neither proves nor disproves the failure.
   - The plan should add a test that hard-deletes an influencer who has a DONE application and a PENDING Meta request.

3. **Cascade delete touching Firebase before the database commits (INFERENCE) still holds.**
   - `deleteByAppliedOpportunityId` is a derived delete without `@Modifying` (`AppliedOpportunityStatusHistoryRepository.java:81`), so it doesn't flush immediately.
   - `userRepository.delete(user)` and `taskRepository.save(task)` queue their writes for the commit.
   - So the user-row delete, and any foreign-key error on `pending_data_deletion_request`, only hits the database at commit, after Firebase Auth, Storage and Firestore have already been deleted.
   - For influencers with several applications, some application deletes may be written earlier, when a later query in the loop forces a flush. That doesn't change the conclusion for the user row.

4. **Retention periods (HYPOTHESIS): unchanged.** They're a legal decision and need counsel.

No other corrections.
