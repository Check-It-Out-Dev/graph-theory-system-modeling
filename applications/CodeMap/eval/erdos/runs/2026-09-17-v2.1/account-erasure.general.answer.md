I've finished the investigation and am writing up the answer. No files were changed and no tests were run.

# Problem

An influencer wants their account erased. The code has **six separate ways to "delete" an account**, and they all do different things. None of them erases everything. Some do almost nothing but still say they worked. And the one record the law clearly needs us to keep, proof that the user gave consent, is exactly what the hard-delete paths end up throwing away.

The five most serious problems:

1. **"Soft delete" doesn't erase an influencer's data, and nothing ever finishes the job.** For influencers, `archiveUser` only clears two email-verification timestamps. Name, phone, email, NIP, profile picture, addresses, Instagram connection data and uploaded files all stay. The account goes to `TO_BE_DELETED`, a state that can be reversed. No scheduled job ever deletes those accounts later. The frontend comment saying the deferred cron "performs the RODO cascade later" is wrong.
2. **Instagram (Meta) callbacks probably fail silently.** The code saves the connection status `DISCONNECTED`, but the database constraint only allows `CONNECTED`, `EXPIRED` and `REVOKED`, and no migration adds the new value. The transaction would then roll back. The controller catches the error and still sends Meta HTTP 200 with a made-up confirmation code. The tests mock the repositories, so they can't catch this. Even when it works, the callback only archives, so for influencers it erases almost nothing.
3. **The hard-delete paths lose consent proof while personal data stays behind.** `consent_record.user_id` is `ON DELETE SET NULL`. Once the user is deleted, their consent records look exactly like anonymous cookie consents. The weekly cleanup job deletes those after a year, but until then the IP address, user agent and proof data sit there with nothing linking them to anyone.
4. **File deletion targets the wrong folder.** The cascade only deletes `users/{uid}/`. Uploads are actually stored under `content/{uid}/` and Instagram profile pictures under `profile-pictures/{uid}/`. The `file_uploads` rows are never deleted either.
5. **The hard-delete paths skip or break on parts of the database.**
   - `/auth/delete-account` and `deletePermanently` call `userRepository.delete` without handling linked records. Any influencer who ever applied to an opportunity would likely hit a constraint error. `/auth/delete-account` also ignores the eligibility check entirely.
   - The admin cascade handles applications, but would likely fail at commit on `pending_data_deletion_request` (a foreign key with no delete rule). By then Firebase Auth, Firestore and Storage have already been deleted, and that can't be undone.

# Where it lives today

## Modules and files

| Area | File(s) | Role |
|---|---|---|
| Eligibility and blockers | `backend/.../user/UserAccountOrchestrator.java`, `user/dto/Deletion*.java` | `checkDeletionEligibilityForUser`, soft-delete blockers (influencer: active or pending applications; company: active opportunities or ones with applications; last admin), `archiveUser` |
| User soft and permanent delete | `backend/.../user/UserService.java:338-390`, `user/UserController.java:330-380, 489-524` | `DELETE /users/{ids}` archives; `DELETE /users/delete-permanently/{ids}` (admin) does a plain JPA delete plus Firebase Auth |
| Password-confirmed self-delete | `backend/.../auth/FirebaseAuthProxyController.java:550-618`, `auth/service/FirebaseAuthProxyService.java:805-907` | Hard delete by email from the JWT, then Firebase Auth delete. No eligibility check, no file or Firestore cleanup |
| Deferred deletion | `auth/cron/DeferredDeletionCronJob.java`, `auth/entity/PendingDataDeletionRequest.java`, `auth/repository/PendingDataDeletionRequestRepository.java`, `db/changelog/2026/03/11-03-2026-create-pending-data-deletion-request.sql` | Runs daily on `PENDING` requests (created only by the Meta callback) and calls `archiveUser` |
| Meta callbacks | `auth/controller/InstagramCallbackController.java`, `auth/service/InstagramDataDeletionService.java`, `auth/service/InstagramDeauthorizationService.java`, `auth/service/MetaSignedRequestService.java` | Data deletion: immediate archive or deferred. Deauthorize: verified-email users get disconnected plus an email; unverified users get archived |
| Admin cascade | `admin/cascade/AdminCascadeDeleteServiceImpl.java`, `CascadeDeleteTask.java`, `dto/ArchiveEnvelope.java` | Deletes PostgreSQL rows, then Firestore, then Storage, then Firebase Auth, with a retry task. The archive pieces are never called |
| Consent | `legal/LegalConsentService.java`, `legal/ConsentRecord*.java`, `legal/AnonymousConsentCleanupCronJob.java`, `db/changelog/2026/03/09-03-2026-consent-module-tables.sql`, `common/001-schema/006-consent-management-tables.sql` | `consent_record` (SET NULL), `user_consent` (CASCADE), cleanup of rows with `user_id IS NULL` older than a year, archiving of users with no consents |
| Storage | `storage/service/FirebaseStorageService.java`, `SignedUrlService.java:65,313`, `ProfilePictureProxyService.java:95`, `common/001-schema/005-file-uploads-table.sql` | Path patterns, `deleteUserDirectory`, `file_uploads` (user_id is a text column with no foreign key) |
| Caches and side stores | `auth/cache/RedisUserCache.java` (5-minute TTL), `common/security/GeoLocationGdprService.java:89`, `common/ratelimit/GdprCompliantRateLimiterService.java:319`, `auth/stepup/StepUpAuthService.java` | The geo and rate-limit erase methods exist but only admin/privacy controllers call them |
| Other personal data in the database | `common/001-schema/002-tables.sql` (`support_ticket`, `search_query_log`, `applied_opportunity_status_history.changed_by_firebase_id`, `user_social_connection`), `2026/01/2026-01-create-notifications-table.sql`, `notification/ActorSnapshot.java` | Name and avatar copied into other users' notifications; ticket email and IP; search log IP |
| Frontend | `frontend/src/app/feature/profile/profile-view.component.ts:208-261`, `core/user/user.service.ts:68-85`, `feature/profile/delete-blockers-dialog.component.ts`, `core/admin/cascade-delete.service.ts`, `app.routes.ts` | Eligibility check in the browser, then `DELETE /users/{id}`, then sign-out. No `/deletion-status` route. The admin service only wires the partnership-opportunity cascade |

## The flows that matter

- **A. Influencer self-delete (frontend):** eligibility check → blockers dialog (dead end) or confirmation → `UserService.delete` → `archiveUser` → `TO_BE_DELETED`, token version bumped, cache evicted. Nothing happens after that.
- **B. `/auth/delete-account`:** password check → delete user by email → Firebase `deleteUser`. Consents lose their user link, files, Firestore data and notification snapshots stay.
- **C. Meta data deletion:** set connection to `DISCONNECTED` (probably rejected by the database) → eligible: archive and mark the request `COMPLETED`; blocked: `TO_BE_DELETED` plus a `PENDING` request → daily cron archives once blockers clear. Only the Firestore `instagramUsers/{uid}` document is deleted.
- **D. Meta deauthorize:** verified email: disconnect, delete the Firestore document, send email. Unverified: disconnect and archive. The comment says this "anonymizes PII", which isn't true for influencers.
- **E. Admin permanent delete:** plain JPA delete plus Firebase Auth.
- **F. Admin cascade:** applications, history and content → user row → (still inside the database transaction) Firestore Instagram and TOTP data, `users/{uid}/` in Storage, Firebase Auth → commit.

# Proposed change

**One erasure pipeline: every entry point records a request, access is cut immediately, and the actual erasure runs as an idempotent, resumable task that keeps a minimal, pseudonymous set of records the law requires.**

1. **A single `AccountErasureService`** is the only code allowed to erase a person. Flows A–F, the no-consent archiving job and the deauthorize unverified-user branch all call `requestErasure(user, source, actor)`. `deletePermanently`, `/auth/delete-account` and `forceDeleteUser` become thin wrappers around it, or are removed. This stops the paths from drifting apart again.

2. **Phase 1: restrict (synchronous, one transaction).**
   - Create an `erasure_request` row. It generalizes `pending_data_deletion_request`: `subject_ref` (a keyed hash of the Firebase UID), a nullable `user_id` with `ON DELETE SET NULL`, source, confirmation code, status, `due_at`, and blockers as JSON with no names or emails in it.
   - Set status to `TO_BE_DELETED`, bump the token version, evict the cache.
   - After commit: revoke Firebase refresh tokens and disable the Firebase user, and delete Instagram tokens from Firestore.
   - Remove `TO_BE_DELETED → ACTIVE/IN_VALIDATION` unless the request is explicitly cancelled inside a grace window, and record the cancellation.
   - Minimize data at once, even when blockers exist: clear phone, addresses, avatar, social connection fields (display name, email, picture, follower count), and stop Instagram sync.

   *Why:* Art. 17 plus Art. 18. Access and processing stop straight away, even when a live collaboration delays full erasure.

3. **Blockers only delay erasure of the collaboration records themselves, never the whole account.** Today a blocked Meta request locks the influencer out, so they can't finish the collaboration, and the request may stay pending forever (inference). Proposal:
   - Add a hard deadline (`due_at`, for example one month, which is the Art. 12(3) response period).
   - Notify the counterpart company when a request comes in.
   - Past the deadline, send the task to an admin queue that has to resolve it (cancel the application or keep a pseudonymized copy). It must never just keep waiting.
   - Enforce the same blocker check on the server for every path, including `/auth/delete-account`.

4. **Phase 2: execute (`ErasureExecutor`, idempotent, ShedLock cron plus manual retry).** Each step has its own status, reusing `CascadeDeleteTask` / `SystemStatus`. The database step commits **before** any external step, which fixes the ordering problem in the admin cascade.
   - **PostgreSQL (one transaction):**
     - Delete: addresses, social connections, preferences, `user_consent`/`user_current_consent`, notifications *received*, `file_uploads` rows (once their blobs are listed in the task), `search_query_log` rows for the user (or null out `user_id` and IP).
     - Anonymize: support tickets (`contact_email`, `ip_address`, description, or delete them), `changed_by_firebase_id` in status history, the actor name and avatar inside *other users'* notification snapshots (`snapshot->'actor'->>'id' = userId` → "Deleted user").
     - Applications and content: delete, unless a retention rule applies (see 5).
     - Consent records: pseudonymize, don't delete (see 5).
     - User row: delete.
   - **External, after commit, each retried on its own:** Storage prefixes `content/{uid}/`, `users/{uid}/`, `profile-pictures/{uid}/`, plus each `file_uploads.file_path` and the URLs in `applied_opportunity_content.urls`. Firestore `instagramUsers/{uid}`, `totpSecrets/{uid}` and any legacy `socialConnections/{uid}_*`. Firebase Auth. Redis: `user_cache:{uid}`, `geo:travel:{userId}:*`, rate-limit `user:{uid}`, step-up keys.
   - **Finish:** request `COMPLETED`, `user_id` null, only `subject_ref`, timestamps, per-system statuses and confirmation code kept. The Meta status URL keeps working.

5. **What we keep because the law requires it (a counsel decision, made explicit in code).** A `RetentionPolicy` component decides, per record type, whether to keep it and until when:
   - **Consent proof (Art. 7(1), Art. 5(2)):** keep `document_id`, timestamp, source and document hash, plus `subject_ref` and `retain_until`. Drop or truncate IP, user agent and screen coordinates, as counsel decides. Add a column so these rows are no longer mistaken for anonymous cookie consents, and point the anonymous cleanup at that column instead of `user_id IS NULL`. Retained rows get their own expiry job.
   - **Paid collaborations** (`TO_BE_PAID`, `DONE` with cash pay): keep the financial facts (amounts, dates, opportunity ID) with the influencer replaced by `subject_ref` until the accounting retention period ends. The period needs legal confirmation (hypothesis: 5 years under Polish accounting/tax rules).
   - **Erasure evidence:** the `erasure_request` row with no personal data.
   - **Archive:** use `ArchiveEnvelope` / `uploadArchive` only if counsel wants a restricted-access archive. If so, it needs its own bucket, encryption, an expiry lifecycle and an auditable reader. Otherwise delete the dead code, because an archive of deleted data is itself data that survives.

6. **Meta specifics.** Add a migration that allows `DISCONNECTED`. Only return a confirmation code if a real `erasure_request` row exists. On failure, create the request in a separate transaction first and then retry, instead of returning a made-up code. On deauthorize for verified users, delete the data obtained from Instagram (avatar blob, connection fields) even when the account stays.

7. **Frontend.** A public `/deletion-status?code=` page. A "request deletion" flow that works with blockers ("scheduled, completes by X; these collaborations will be closed or pseudonymized"). The admin erasure-task list and retry wired to the API. The `user.service.ts` comment corrected.

# Plan

1. **Schema migrations** (new files under `backend/src/main/resources/db/changelog/2026/09/`, registered in `changelog.xml`):
   - Allow `DISCONNECTED` in `user_social_connection`.
   - Create `erasure_request`, with data migrated from `pending_data_deletion_request` and a unique partial index for one open request per user.
   - Change `pending_data_deletion_request.user_id` to `ON DELETE SET NULL`, or drop the table after migrating.
   - Add `consent_record.subject_ref`, `retain_until` and `retention_reason`.
   - Add `applied_opportunity.influencer_subject_ref`; `influencer_id` becomes nullable only if retention applies. This removes the NOT NULL + `ON DELETE SET NULL` contradiction.
   - Add `cascade_delete_task.erasure_request_id`.
2. **Retention policy:** new `backend/.../erasure/RetentionPolicy.java` plus config in `application.yml` / `application-prod.yml`, with values signed off by counsel.
3. **Request side:** new `erasure/AccountErasureService.java` and `erasure/ErasureRequest*.java`. Refactor `UserAccountOrchestrator.archiveUser` to the "restrict and minimize" step for every user type. Tighten `AccountStatus.canTransitionTo` and `UserService` status updates around line 717.
4. **Execution side:** new `erasure/ErasureExecutor.java` plus a cron. Move logic out of `AdminCascadeDeleteServiceImpl` so it wraps the executor and the database commit comes before external calls. Add `FirebaseStorageService.deleteUserData(uid)` covering all three prefixes plus explicit paths, and fix `countUserFiles` / `userDirectoryExists` the same way. Add a `FileTrackingService` method to list and delete by user. Add `NotificationRepository` snapshot scrubbing. Add support ticket and search log anonymization in their repositories. Call `GeoLocationGdprService.deleteUserLocationData`, `GdprCompliantRateLimiterService.deleteUserRateLimitData`, `StepUpAuthService` and `UserCacheService.evict`.
5. **Consent handling:** in `legal/LegalConsentService.java` and `ConsentRecordRepository.java`, add `pseudonymizeForErasure(userId, subjectRef)`. Change `deleteAnonymousRecordsBefore` to exclude retained rows. Add a new `RetainedConsentExpiryCronJob`. Route `archiveUsersWithNoConsents` to `requestErasure`.
6. **Point every entry point at the new service:**
   - `UserService.delete` / `deletePermanently` and `UserController`.
   - `FirebaseAuthProxyService.deleteAccount`: look the user up by Firebase UID, not email, and enforce eligibility.
   - `InstagramDataDeletionService`, `InstagramDeauthorizationService` (unverified branch plus Instagram-data scrub for verified users), `InstagramCallbackController`: no made-up codes, status from `erasure_request`.
   - `DeferredDeletionCronJob`: replace it with the executor, with deadline escalation.
7. **Logging:** remove raw email and blocker JSON from logs in `LegalConsentService.java:547`, `InstagramDataDeletionService.java:179`, `UserAccountOrchestrator.generateDeletionSummary`, and the free-text `reason` in `FirebaseAuthProxyController.java:587`. Separately, confirm log retention with ops (logs are outside the code).
8. **Frontend:**
   - `app.routes.ts` plus a new `feature/deletion-status/` page.
   - `feature/profile/profile-view.component.ts`, `delete-blockers-dialog.component.ts` and `core/user/user.service.ts` for the scheduled-deletion flow and the corrected comment.
   - Admin erasure tasks: `core/admin/cascade-delete.service.ts` plus an admin view.
   - Regenerate `src/app/api/**` from the updated `docs/openapi/openapi.json`.
   - `assets/i18n/en.json` and `pl.json`.
9. **Remove or guard dead paths:** `ArchiveEnvelope` / `uploadArchive` (per the step 2 decision), `findOrphanedFirebaseUsers` (returns an empty list).

# Risks and invariants

| Invariant | Guarding test (new unless noted) |
|---|---|
| **I1.** Once erasure completes, no row outside the retention allowlist holds the user's ID, Firebase UID, email, phone or name. | Testcontainers `ErasureCompleteness_IntegrationTest`: seed an influencer with every related table filled (addresses, connection, prefs, applications, content, history, tickets, search log, notifications sent and received, consents, `file_uploads`), run the executor, then query every column in `information_schema` for the seeded values. |
| **I2.** The database commits before any irreversible external delete, and each external step is idempotent. | `ErasureExecutor_OrderingUnitTest` (a failing commit means Firebase is never called); an integration test that runs the executor twice. |
| **I3.** Meta callbacks never report success unless an `erasure_request` exists; `DISCONNECTED` saves. | `InstagramCallback_IntegrationTest` on real Postgres. The existing `InstagramDataDeletionServiceUnitTest` and `InstagramDeauthorizationServiceUnitTest` mock the repositories and didn't catch the constraint problem. |
| **I4.** Retained consent records survive the anonymous-cleanup job and expire on `retain_until`. | `LegalConsentService_Retention_IntegrationTest`. |
| **I5.** Storage cleanup covers every upload path prefix. | `FirebaseStorageService_ErasureUnitTest` driven by the `storage-path-pattern` value and the `ProfilePictureProxyService` format, so a new prefix breaks the test. |
| **I6.** Blockers can't delay erasure past `due_at` without an admin task; a user who is `TO_BE_DELETED` can't be reactivated outside the grace window. | `DeferredDeletionCronJobUnitTest` (existing, to be extended) → `ErasureDeadlineUnitTest`; `AccountStatus` transition unit test. |
| **I7.** The last admin is protected on every path. | The existing `UserService_Delete_IntegrationTest` (extend it to cover `requestErasure`). |
| **I8.** Session invalidated and cache evicted on request. | Existing `softDeleteIncrementsTokenVersion` in `UserService_Delete_IntegrationTest`; `UserCacheServiceUnitTest`. |
| **I9.** Frontend: blocked users can schedule deletion; the status page shows the backend status. | `profile-view.component.spec.ts` (update); a new spec for the status page; BDD `profile-non-critical-consolidated.feature`. |

**Other risks:**
- Scrubbing notification snapshots changes what companies see ("Deleted user"). Product needs to accept that.
- Keeping pseudonymized financial records means `subject_ref` is still personal data (it's pseudonymous, not anonymous). Its key needs access control and rotation rules.
- Migrating live `TO_BE_DELETED` users: existing ones need a backfill of `erasure_request` rows. Decide whether to run them through the pipeline now; this is likely overdue.
- CDN caching: profile pictures are served with `Cache-Control: public, max-age=31536000`, so deleted avatars can stay in caches. Use short cache lifetimes or unguessable paths from now on.
- Changing the delete-account and callback contracts means regenerating the OpenAPI client, and any external consumers need to be told.

# Evidence

| # | Claim | Status | Source |
|---|---|---|---|
| 1 | Influencer archive only clears verification timestamps; company archive sets the names to "N/A" and nulls the phone | FACT | `UserAccountOrchestrator.java:244-281` |
| 2 | Archive sets `TO_BE_DELETED`, bumps token version, evicts cache | FACT | `UserAccountOrchestrator.java:71-86` |
| 3 | `TO_BE_DELETED` can go back to `ACTIVE`/`IN_VALIDATION` | FACT | `AccountStatus.java:74-75` |
| 4 | Nothing moves `TO_BE_DELETED` users to deleted on a schedule | INFERENCE | grep for `TO_BE_DELETED` across `backend/src/main/java` found no scheduled purge |
| 5 | The deferred cron only processes `PENDING` requests and only archives | FACT | `DeferredDeletionCronJob.java:56-109` |
| 6 | Deferred requests are only created by the Meta callback | INFERENCE | grep: `PendingDataDeletionRequest.builder` only in `InstagramDataDeletionService.java` |
| 7 | Frontend comment claims the cron performs the cascade | FACT | `frontend/src/app/core/user/user.service.ts:79-81` |
| 8 | Frontend eligibility check is client-side; DELETE doesn't re-check | FACT (comment) | `profile-view.component.ts:199-207` |
| 9 | Influencer archive still throws on active opportunities | FACT | `UserAccountOrchestrator.java:251-254` |
| 10 | Code sets `DISCONNECTED`; DB CHECK allows only CONNECTED/EXPIRED/REVOKED; no migration adds it; `ddl-auto: validate` | FACT | `ConnectionStatus.java:23`; `002-tables.sql:120`; grep of `db/`; `application.yml:45` |
| 11 | So Meta callback transactions roll back and the controller returns a made-up code with 200 | INFERENCE | 10 + `InstagramDataDeletionService.java:58,104-105` + `InstagramCallbackController.java:66-73` |
| 12 | Meta callback tests are unit tests with mocks | INFERENCE | test paths under `unit/service/` only |
| 13 | Blocked Meta request locks the user out, so collaborations may never finish | INFERENCE | `InstagramDataDeletionService.java:160-166`; `RedisUserCache.java:45-50` |
| 14 | Data-deletion callback only deletes the Firestore `instagramUsers` document | FACT | `InstagramDataDeletionService.java:122`; `FirestoreService.java:406-415` |
| 15 | Deauthorize comment claims archive "anonymizes PII" | FACT | `InstagramDeauthorizationService.java:111` |
| 16 | `findByUserId` returns Optional but `user_id` isn't unique | FACT / INFERENCE (a second request would throw) | `PendingDataDeletionRequestRepository.java:20`; migration SQL has no unique on `user_id` |
| 17 | `pending_data_deletion_request.user_id` foreign key has no delete action | FACT | `11-03-2026-create-pending-data-deletion-request.sql:17` |
| 18 | `/auth/delete-account` hard-deletes by email, skips eligibility, files and Firestore | FACT | `FirebaseAuthProxyService.java:856-868` |
| 19 | `applied_opportunity.influencer_id` is NOT NULL with `ON DELETE SET NULL` | FACT | `002-tables.sql:187` |
| 20 | So a plain user delete fails for influencers with applications | INFERENCE | 18/19 + `UserService.java:379` |
| 21 | Admin cascade does external deletes inside `@Transactional` before commit | FACT | `AdminCascadeDeleteServiceImpl.java:243-364` |
| 22 | Commit failure (e.g. 17) after Firebase deletion leaves the database intact but Firebase gone | INFERENCE | 17 + 21 |
| 23 | On database failure the `FAILED` task save is rolled back with the rest | INFERENCE | `AdminCascadeDeleteServiceImpl.java:286-292` (same transaction) |
| 24 | Cascade deletes only `users/{uid}/` | FACT | `FirebaseStorageService.java:42-53` |
| 25 | Uploads go to `content/{userId}/`; profile pictures to `profile-pictures/{uid}/` with a one-year public cache | FACT | `application-prod.yml:230`; `SignedUrlService.java:65,313-321`; `ProfilePictureProxyService.java:95-102` |
| 26 | `file_uploads.user_id` is text with no foreign key; never cleaned on delete | FACT / INFERENCE | `005-file-uploads-table.sql:13`; no call in the delete paths |
| 27 | Archive envelope and `uploadArchive` exist but nothing calls them; `archiveUrl` is never set | INFERENCE | grep `uploadArchive` / `archiveUrl` |
| 28 | `consent_record.user_id` is `ON DELETE SET NULL`; anonymous cleanup deletes rows with null `user_id` older than a year; rows keep IP, user agent and proof | FACT | `09-03-2026-consent-module-tables.sql:36-46`; `ConsentRecordRepository.java:69-70`; `LegalConsentService.java:570-576` |
| 29 | So a deleted user's consent proof is lost while IP and user agent stay until then | INFERENCE | 28 |
| 30 | `user_consent` and `user_current_consent` cascade on delete | FACT | `006-consent-management-tables.sql:41,67` |
| 31 | Notification snapshots store actor name and avatar "if user deletes account"; `influencer_id` has no foreign key | FACT | `ActorSnapshot.java:15-18,60-75`; `2026-01-create-notifications-table.sql:37,42` |
| 32 | `support_ticket` keeps `contact_email` and `ip_address` (SET NULL); `search_query_log` keeps IP; status history keeps `changed_by_firebase_id` | FACT | `002-tables.sql:219-220,272-288,380-387` |
| 33 | Geo and rate-limit erase methods exist but only controllers call them | FACT / INFERENCE | `GeoLocationGdprController.java:72-79`; `RateLimitPrivacyController.java:58`; grep of callers |
| 34 | User cache TTL is 5 minutes | FACT | `RedisUserCache.java:35` |
| 35 | Frontend has no `/deletion-status` route although the backend sends Meta that URL | FACT | `app.routes.ts` grep; `InstagramDataDeletionService.java:186` |
| 36 | Frontend admin cascade service only wires the partnership-opportunity cascade | FACT | `frontend/src/app/core/admin/cascade-delete.service.ts:22-32` |
| 37 | Existing integration test only checks company anonymization and the `TO_BE_DELETED` status | FACT | `UserService_Delete_IntegrationTest.java:82-102` |
| 38 | Personal data in logs (email in no-consent archive log; blockers JSON; eligibility summary with email) | FACT | `LegalConsentService.java:547-549`; `InstagramDataDeletionService.java:179`; `UserAccountOrchestrator.java:293` |
| 39 | Legacy Firestore `socialConnections` documents may still exist from before deprecation | HYPOTHESIS | `FirestoreService.java:196-224` (commented-out writer) |
| 40 | Consent proof and paid-collaboration records must be kept for set periods (Art. 7(1)/5(2); Polish accounting ~5 years) | HYPOTHESIS (legal) | Needs counsel sign-off; not in code |
