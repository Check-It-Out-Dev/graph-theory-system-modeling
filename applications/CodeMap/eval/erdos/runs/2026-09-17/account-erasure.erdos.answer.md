## Problem
An influencer asks for their account to be erased. Today the account is marked `TO_BE_DELETED` and the sessions are revoked. Most personal data stays in Postgres, Firestore, Firebase Storage and Firebase Auth, and nothing in the code finishes the job later. "Done" means three things. One pipeline runs whatever started the erasure (the user, Meta, or an admin). It removes or pseudonymises every personal-data store, and each step can be retried and audited. It keeps only what the law requires (invoices, proof of consent), with the identity stripped out.

## Where it lives today
**Subsystems:** [5] account deletion & Instagram sync, [10] user identity, [11] consent & billing, [6] user cache & Firestore, [3] storage, [15] support tickets. On the frontend: [176] profile (self-delete), [171]/[176] admin cascade delete, and [177] the generated client.

**Entry points and flows (backend unless marked)**
1. **User deletes their own account.** FE `src/app/feature/profile/profile-view.component.ts:208-261` calls `GET /users/me/deletion-eligibility`, opens a dialog, then calls `DELETE /users/{id}`. BE `user/UserController.java:330-342` → `user/UserService.java:340-354` `delete()` → `UserAccountOrchestrator.archiveUser()`. The eligibility check only runs in the browser: the FE comment at `core/user/user.service.ts:68-73` says so.
2. **Archival.** `user/UserAccountOrchestrator.java:55-87`, per user type:
   - **Influencer** (`:244-264`): pending applications are deleted and the two email-verification timestamps are cleared. Nothing else.
   - **Company** (`:266-281`): first and last name become "N/A" and the phone number is cleared.
   - **Everyone:** status becomes `TO_BE_DELETED`, the token version is bumped, and the user is evicted from the cache.
3. **Eligibility and blockers** (`UserAccountOrchestrator.java:92-240`):
   - Influencer: active applications **and** pending (`APPLIED`) applications both block.
   - Company: active opportunities, or opportunities with applications.
   - Admin: the last admin cannot be deleted.
   - The support-ticket check and every permanent-deletion blocker are empty placeholders.
4. **Meta data-deletion callback.** `auth/controller/InstagramCallbackController.java:59-75` → `auth/service/InstagramDataDeletionService.java`:
   - The connection is set to `DISCONNECTED`.
   - Case A (no blockers): `archiveUser` runs and a `COMPLETED` `PendingDataDeletionRequest` is saved.
   - Case B (blockers): the user becomes `TO_BE_DELETED` and a `PENDING` request is saved.
   - Afterwards it makes one best-effort call to `firestoreService.deleteInstagramUserData`.
   - If anything throws, the controller answers 200 with a random confirmation code and saves nothing.
5. **Deferred deletion.** `auth/cron/DeferredDeletionCronJob.java` runs daily at 03:00 under ShedLock. For each `PENDING` request it re-checks eligibility and calls `archiveUser`, which is the same partial anonymisation.
6. **Meta deauthorization.** `auth/service/InstagramDeauthorizationService.java`:
   - Verified email: disconnect, delete the Firestore `instagramUsers` document, send an email.
   - Unverified email: `archiveUser` inside a try/catch, inside an `@Transactional` method.
7. **Admin permanent delete.** `UserService.deletePermanently` (`:356-390`) evicts the cache, calls `repository.delete(user)`, then `firebaseService.deleteUserGraceful`. It cleans up nothing in Firestore or Storage.
8. **Admin cascade delete.** `admin/cascade/AdminCascadeDeleteServiceImpl.java:245-377`, in this order:
   - Postgres: status history → applied opportunities → partnership opportunities → the user row.
   - Then, outside the transaction and each step tracked in `CascadeDeleteTask` with retry: Firestore `instagramUsers`, Firestore `totpSecrets` plus its audit log, Storage `users/{uid}/`, Firebase Auth.
   - Finally, cache eviction.
   - FE clients: `core/admin/cascade-delete.service.ts`, with fixtures in [173]/[178].
9. **Deletion status page.** Meta sends people to `frontendUrl + "/deletion-status?code="`. The backend serves `GET /auth/instagram/deletion-status`.

**Where personal data survives today (the gaps)**

| # | Gap | Where |
|---|---|---|
| G1 | For an influencer, erasure keeps the email, first and last name, display name, profile picture, phone, admin note, `lastVerifiedEmail` and `firebaseUserId`. | `UserAccountOrchestrator:244-264`, fields in `User.java` |
| G2 | Nothing ever moves a user from `TO_BE_DELETED` to `DELETED`. There is no purge job, and `deletedAt` is never set. | grep of `AccountStatus.DELETED` and `setDeletedAt` |
| G3 | Nothing is deleted from Firebase Auth on self-delete, the Meta callbacks or the cron. Only the two admin paths delete there. | `UserService:384`, `AdminCascade:335` |
| G4 | Every upload path is missed. Uploads go to `content/{uid}/` (`application*.yml` `storage-path-pattern`) and `profile-pictures/{uid}/` (`ProfilePictureProxyService:95`), but cascade delete only removes `users/{uid}/` (`FirebaseStorageService:42-53`). The `file_uploads.user_id` column holds the Firebase UID with no foreign key, so those rows also survive a hard delete. | as cited |
| G5 | Social-connection rows keep `socialUserId`, `email`, `displayName`, `profileUrl`, `profilePictureUrl` and follower count. The Meta callback only sets `DISCONNECTED`. | `UserSocialConnection.java`, `InstagramDataDeletionService:104` |
| G6 | Self-delete and deauthorization never touch the TOTP secrets and audit log in Firestore. | `TotpFirestoreService:405` is called only from the cascade |
| G7 | Support tickets keep `contactEmail` and `ipAddress` (`ON DELETE SET NULL`). `search_query_log` keeps the query text. `consent_record` keeps IP and user agent (`SET NULL`). The older `user_consent` table is deleted with the user (`CASCADE`), which destroys proof of consent. | changelog `002-tables.sql:274,384`, `09-03-2026-consent-module-tables.sql:36`, `006-consent-management-tables.sql:41` |
| G8 | Hard delete clashes with the schema. `pending_data_deletion_request.user_id` has no `ON DELETE` rule, so the delete fails with a foreign-key error for anyone who ever used the Meta callback. `applied_opportunity.influencer_id` is `NOT NULL` with `ON DELETE SET NULL`. | `11-03-2026-create-pending-data-deletion-request.sql:17`, `002-tables.sql:187` |
| G9 | Hard delete destroys records the law requires. `invoice_record`, `billing_period`, `subscription_event` and `company_subscription` are all `ON DELETE CASCADE`. This affects companies, but the same pipeline serves them. | `22-03-2026-subscription-tables.sql:45-114` |
| G10 | A failed Meta callback returns a fake confirmation code and persists no request, so the erasure is silently lost. | `InstagramCallbackController:66-73` |
| G11 | A deferred request can stay stuck forever. Pending applications block deletion, but the user in `TO_BE_DELETED` can no longer withdraw them, and there is no deadline. | `UserAccountOrchestrator:151-165`, `InstagramDataDeletionService:160-166` |
| G12 | The frontend has no `/deletion-status` route, so the status link Meta shows the user leads nowhere. | grep over the frontend: only the generated API path exists |
| G13 | The `DELETE /users/{id}` endpoint does not re-check eligibility on the server. | FE comment `user.service.ts:71-72` |

## Proposed change
**Chosen design: a two-stage erasure pipeline that keeps a pseudonymous tombstone.**

**Why not the alternative:** routing every request to the admin cascade (a hard delete). That would destroy invoices and proof of consent (G7, G9), fail on the foreign keys (G8), and break the company-side history of collaborations.

**Stage 1 — request.** Every entry point (self-delete, the Meta data-deletion callback, deauthorization of an unverified user, an admin request) calls one `ErasureService.request(user, source)`.
- It checks eligibility on the server.
- It sets `TO_BE_DELETED`, bumps the token version, disables the Firebase Auth user and strips its claims, and evicts the cache.
- It persists an `ErasureRequest` with a confirmation code, the source, and a legal deadline (`requestedAt + 30 days`).
- The request is saved even when a later step fails, which fixes G10.
- Pending applications are withdrawn automatically instead of blocking. Only active collaborations defer the erasure, which fixes G11.

**Stage 2 — execute.** `DeferredDeletionCronJob` becomes the executor. Once no blockers remain, or the deadline passes and an admin has been alerted, it runs an ordered list of `PersonalDataEraser` steps. Each step is idempotent and has its own status and retry, generalising `CascadeDeleteTask`.

*Postgres, in one transaction:*
1. Pseudonymise the `User` row:
   - `email` → `deleted-<id>@invalid`; names, `name`, phone, `profilePicture`, `noteFromAdmin` and `lastVerifiedEmail` → null.
   - `firebaseUserId` → `deleted-<uuid>`, because the column is `NOT NULL UNIQUE`.
   - Status → `DELETED`, set `deletedAt`.
2. Delete addresses, preferences, social connections and notifications.
3. Redact support tickets (`contactEmail` and `ipAddress` → null) and null out `search_query_log.user_id` and its query text.
4. Consent:
   - Keep `consent_record` and `user_consent` rows as proof of consent, pointing at the tombstone.
   - Truncate the IP to /24 and drop the user agent.
5. Applied opportunities: keep the row for the company's history, linked to the tombstone; delete submitted content and its files. Whether this is allowed is a legal question, flagged below.
6. Billing rows are never deleted.

*External stores, after the commit:*
7. Firestore: `instagramUsers`, `totpSecrets` with its `auditLog`, and the social-connections collection.
8. Storage: `users/{uid}/`, `content/{uid}/` and `profile-pictures/{uid}/`, plus the matching `file_uploads` rows.
9. Firebase Auth: delete the user, last.
10. Cache: evict.

The `ErasureRequest` keeps the confirmation code, the status, the timestamps and the per-step outcomes, and holds no personal data. The admin cascade stays for exceptional hard deletes, but uses the same eraser steps, which fixes G4 there. Schema changes make the retention rules safe (G8, G9).

## Plan
1. **Schema migration** (new Liquibase changesets under `db/changelog/2026/09/`):
   - `pending_data_deletion_request` → rename or extend it to hold the erasure request: add `source`, `deadline_at` and per-step status as JSONB; set `user_id` to `ON DELETE SET NULL`.
   - `invoice_record`, `billing_period`, `subscription_event` → `ON DELETE RESTRICT`.
   - Make `applied_opportunity.influencer_id` consistent: keep `NOT NULL` and switch to `RESTRICT`, since the tombstone stays.
   - Add `consent_record.ip_truncated`, or reuse the existing column.
   - Nothing else changes behaviour yet, so the system keeps working.
2. **Eraser steps** (new package `backend/.../platform/user/erasure/`): the `PersonalDataEraser` interface and `ErasureService`, plus one step per store:
   - `UserTombstoneEraser` and `SatelliteRowsEraser` (address, preferences, social connections, notifications, `file_uploads`, `search_query_log`).
   - `SupportTicketRedactor` in [15] and `ConsentPseudonymiser` in [11].
   - `FirestoreEraser`, which reuses `FirestoreService.deleteInstagramUserData` and `TotpFirestoreService.deleteTotpData` and adds social-connections cleanup.
   - `StorageEraser`: new `FirebaseStorageService.deleteUserContent(uid)` covering all three prefixes.
   - `FirebaseAuthEraser`, which reuses `deleteUserGraceful`.
3. **Stage 1 in place of `archiveUser`:**
   - `UserAccountOrchestrator.java`: `archiveUser` delegates to `ErasureService.request`.
   - Influencer and company archival keep only the opportunity handling. Pending applications no longer block; they are auto-withdrawn.
   - `UserService.delete` re-checks eligibility on the server and returns 409 with the blockers (G13).
4. **Executor:** rewrite `DeferredDeletionCronJob.java` to process `PENDING` and `PARTIAL` requests. It runs the steps, retries failed external steps, and escalates to admins when a request passes its deadline.
5. **Meta callbacks:**
   - `InstagramDataDeletionService.java`: persist the request first in its own transaction (`REQUIRES_NEW`), then run the rest.
   - `InstagramCallbackController.java:66-73`: on failure, look up that persisted request instead of inventing a code (G10).
   - `InstagramDeauthorizationService.java`: move `archiveUser` out of the swallowed try/catch inside the transaction (see Phase 2).
6. **Admin paths:**
   - `AdminCascadeDeleteServiceImpl.java:297-360` and `retryTask`: use the same eraser steps and add the content and profile-picture prefixes.
   - `UserService.deletePermanently`: route through the cascade, or remove it, because it fails on G8 and skips Firestore and Storage.
7. **Frontend:**
   - Regenerate the client with `openapi:gen` ([177] `api`).
   - Add a public `deletion-status` route and component that calls `InstagramCallbackControllerService` (router in [177], the screen in [176]).
   - `profile-view.component.ts`: handle the 409 blockers response from the server and tell the user about the deadline.
   - Admin cascade screens and `admin-cascade-delete.fixture.ts` / `demo-fixtures.ts`: show the new per-step statuses.
8. **Backfill:** a one-off job that runs Stage 2 for every existing `TO_BE_DELETED` user whose request is older than 30 days (G2). Do a dry-run preview first, using `previewUserDeletion`.

## Risks and invariants
- **Records the law requires stay.** Invoices, billing periods and subscription events are never deleted. Guard: a migration test that a `DELETE` of a user with invoices fails with `RESTRICT`, and a unit test that the eraser never touches billing repositories. The retention period is a legal decision (for example the Polish accounting and VAT rules); confirm it with counsel or the DPO.
- **Proof of consent is kept but no longer identifies the person.** Guard: `ConsentPseudonymiserUnitTest` — the row survives, the IP is truncated, the user agent is null.
- **Idempotency.** Meta retries callbacks, and the cron can overlap with admin actions. Every step must be a no-op on a second run, and the confirmation code must stay stable per user. Guard: extend `InstagramDataDeletionServiceUnitTest` for a duplicate callback, and add a test that runs the cron twice.
- **Order.** Postgres is committed before external deletes, and Firebase Auth goes last, so a failure never leaves a live login without a database row. Guard: an integration test with failure injection on Storage that ends with the request in `PARTIAL`, which a retry completes.
- **Authorization.** Only the user themselves or an admin can request; the last-admin blocker stays. Guard: the existing `UserService` delete permission tests, plus the new 409 test.
- **Unique constraints.** Pseudonymised values must stay unique (`firebase_user_id UNIQUE`) and must free the email so the person can register again. Guard: an integration test that deletes the user and then re-registers with the same email.
- **Sessions.** After Stage 1, any existing token returns 401 or 403 (token version plus disabled Firebase user). Guard: an extension of the Cucumber auth scenarios in [9].
- **Collaboration history.** Company views of applied opportunities must render a tombstone influencer without crashing. Guard: a test on the applied-opportunity DTO mapping with nulled names.
- **Outside the code.** Application logs contain IDs and some emails; `EmailService:338` logs `RecipientEmail`. Log retention is set up outside the repository and needs a DPO decision.

## Evidence
- FACT — Influencer archival only deletes pending applications and clears the verification timestamps; company archival sets names to "N/A" and clears the phone (`UserAccountOrchestrator.java:244-281`).
- FACT — `archiveUser` sets `TO_BE_DELETED`, bumps the token version, saves and evicts the cache (`:55-87`).
- FACT — Pending (`APPLIED`) applications block deletion; the support-ticket and permanent-deletion blockers are placeholders (`:151-165`, `:221-240`).
- FACT — No Java code sets `AccountStatus.DELETED` or calls `setDeletedAt` (grep over `backend/src/main/java`).
- FACT — The Meta callback flow, including Case A and Case B and the post-commit Firestore delete, is as described (`InstagramDataDeletionService.java:58-183`).
- FACT — The controller returns a dummy code on exception without persisting anything (`InstagramCallbackController.java:66-73`).
- FACT — The cron re-checks eligibility and then only calls `archiveUser` (`DeferredDeletionCronJob.java:90-114`).
- FACT — Deauthorization calls `archiveUser` inside a try/catch in an `@Transactional` method (`InstagramDeauthorizationService.java:43,105-126`).
- FACT — The cascade deletes Storage `users/{uid}/` only (`FirebaseStorageService.java:42-53`). Uploads use `content/{userId}/` (`application.yml:537`, `SignedUrlService.java:65,313-321`) and profile pictures use `profile-pictures/{uid}/` (`ProfilePictureProxyService.java:95`).
- FACT — The foreign-key rules in G7, G8 and G9 are as quoted from the Liquibase changesets. `file_uploads.user_id` is a Firebase UID varchar with no foreign key (`005-file-uploads-table.sql:13,37`).
- FACT — `UserService.deletePermanently` does a repository delete plus a Firebase Auth delete only (`UserService.java:356-390`).
- FACT — The frontend deletes through `DELETE /users/{id}` after an eligibility check in the browser (`profile-view.component.ts:208-261`, `user.service.ts:67-85`). No frontend route matches `deletion-status` (grep).
- FACT — The personal-data fields on `User`, `UserSocialConnection`, `SupportTicket`, `ConsentRecord` and `InstagramUserDocument` are as listed (field greps).
- INFERENCE — `DELETE /users/{id}` reaches `UserService.delete`, because `UserController.delete` calls `super.delete(ids)` and `UserService` overrides `delete(Long)`. Verified in Phase 2.
- INFERENCE — A permanent or cascade delete of a user who has a `pending_data_deletion_request` row fails with a foreign-key violation: the constraint has no `ON DELETE` rule and the cascade code never deletes those rows (its repository list has no such repository).
- INFERENCE — In deauthorization Branch A, an exception thrown inside `archiveUser` marks the shared transaction rollback-only (Spring's default `REQUIRED` propagation plus a `RuntimeException`), so the swallowed catch still ends in a rollback.
- HYPOTHESIS — Firestore holds per-user social-connection documents (`SOCIAL_CONNECTIONS_COLLECTION` in `FirestoreService:219`) that no deletion path removes. Checked in Phase 2.
- HYPOTHESIS — Keeping applied-opportunity rows linked to a tombstone is lawful as a legitimate interest or contractual record. This needs legal sign-off.

## Corrections after verification

**What the checks settled**
- **Self-delete reaches the archival code (was INFERENCE, now FACT).** `BaseController.delete` (`common/base/BaseController.java:246-269`) calls `getService().delete(id)` for one ID. For several IDs it calls `BaseService.deleteAll` (`BaseService.java:295-299`), which calls `delete(id)` for each. Both routes end in `UserService.delete` → `archiveUser`.
- **Firestore social-connection documents are never erased (was HYPOTHESIS, now FACT).** `FirestoreService.java:217-220` writes one document per user and platform (`{firebaseUid}_{platform}`). `deleteInstagramUserData` (`:406-431`) deletes only `instagramUsers/{uid}`, and it is the only Firestore delete that any deletion path calls. So these documents survive every path, the admin cascade included. The `FirestoreEraser` in plan step 2 must delete `{uid}_*` in that collection.
- **The cascade cannot handle deletion requests (INFERENCE, stronger now).** Nothing under `platform/admin` refers to `PendingDataDeletionRequest`. Combined with the foreign key that has no `ON DELETE` rule, admin cascade and permanent delete will fail for any user who has ever gone through the Meta callback.
- **Deauthorization rollback (still INFERENCE).** Both methods involved are `@Transactional` (read at `InstagramDeauthorizationService.java:43` and `UserAccountOrchestrator.java:55`). The call goes between two beans, so the proxy applies. The rollback-only outcome follows from Spring's default rules, not from a test run; plan step 5 should prove it with an integration test.

**Corrections to the answer**
1. Add one more place where data survives, **G14: Firestore social-connection documents** (`{firebaseUid}_{platform}`) are never deleted by any path, the admin cascade included. The Evidence line for this becomes a FACT.
2. The claim that `DELETE /users/{id}` reaches `UserService.delete` becomes a FACT, for both single and multiple IDs.
3. The claim that keeping applied-opportunity rows linked to a tombstone is lawful remains a HYPOTHESIS. It needs a ruling from the DPO or legal counsel before plan step 2 ships.

No files were modified.
