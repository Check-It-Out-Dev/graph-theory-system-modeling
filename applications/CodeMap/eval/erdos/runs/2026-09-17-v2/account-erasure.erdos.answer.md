# Erasing an influencer's account: how deletion works today, where personal data survives, and a design that guarantees erasure

## Problem
An influencer asks for their account to be erased. Done means two things. First, every copy of their personal data is removed or anonymised: PostgreSQL, Firestore, Firebase Auth, Firebase/GCS storage, caches and logs. This must hold whichever way deletion starts: in the app, from a Meta callback, by an admin, or by the no-consent cron. Second, only the records the law requires survive: proof of consent, accounting and invoice records, and a trace of the erasure itself. Each has a set retention period and is linked to a record that no longer identifies the person.

## Where it lives today

**Subsystems involved:**
- **[5] Account deletion & Instagram sync** holds most of the deletion code.
- **[11] Subscriptions, payments & consent** holds consent records, invoices and the no-consent cron.
- **[6] Two-factor auth & user cache** holds the user cache and the Firestore services.
- **[3] Rate limits & runtime config** holds storage.
- **[10] User identity & token exchange** holds the `User` entity.
- **Frontend:** [176] has the profile deletion flow, [171] the admin cascade dialog, [177] the generated client.

**Five ways deletion starts, and none of them share code:**

| # | Entry point | Path | What it does |
|---|---|---|---|
| A | User deletes in the app | `frontend/src/app/feature/profile/profile-view.component.ts` → `DELETE /users/{id}` → `backend/.../user/UserService.java` `delete()` → `UserAccountOrchestrator.archiveUser()` | Soft delete: status `TO_BE_DELETED`, token version bumped, cache evicted |
| B | Meta data-deletion callback | `backend/.../auth/controller/InstagramCallbackController.java` `POST /auth/instagram/data-deletion` → `InstagramDataDeletionService` | Archives now if nothing blocks; otherwise queues a `PendingDataDeletionRequest` that `backend/.../auth/cron/DeferredDeletionCronJob.java` re-checks daily at 03:00 |
| C | Meta deauthorize callback | `POST /auth/instagram/deauthorize` → `InstagramDeauthorizationService` | Unverified email: archive. Verified email: disconnect Instagram and send an email |
| D | Admin cascade delete | `backend/.../admin/cascade/AdminCascadeDeleteController.java` → `AdminCascadeDeleteServiceImpl.forceDeleteUser` | Hard delete: PostgreSQL rows, Firestore, storage `users/{uid}/`, Firebase Auth, cache. Failed steps are retried by `OrphanCleanupTask` |
| E | Other hard deletes | `UserService.deletePermanently`; `FirebaseAuthProxyService.deleteAccount`; `LegalConsentService.archiveUsersWithNoConsents` (weekly cron) | The first two delete the `User` row and the Firebase user. The cron calls `archiveUser` |

**Eligibility:** `UserAccountOrchestrator.checkDeletionEligibilityForUser`. For an influencer, active or pending applications block deletion. The support-ticket blocker is only a placeholder, and there are no checks at all for legal holds or retention.

**Where personal data survives today (the evidence is below):**
1. **Nothing ever finishes a soft delete.** No code sets `AccountStatus.DELETED`, and no job deletes `TO_BE_DELETED` users. For an influencer, `archiveUser` only clears the two email-verification timestamps. Email, name, phone, profile picture, addresses, social connections, preferences, notifications, Firestore Instagram and TOTP data, uploaded files and the Firebase Auth user all stay (paths A, B, C and the cron). An admin can even set the account back to active.
2. **Admin cascade deletes the wrong storage folder.** Uploads are written to `content/{userId}/...`, but the cascade deletes `users/{uid}/`. Uploaded content survives even the strongest path.
3. **The in-app flow never queues a deferred deletion.** For an influencer with active work, `archiveUser` throws, so the request just fails. Only the Meta path creates a pending request.
4. **Hard deletes destroy what must be kept, or fail:**
   - `user_consent`, `invoice_record`, `subscription_event` and `billing_period` are set to `ON DELETE CASCADE`, so proof of consent and invoice records disappear with the user.
   - `consent_record` goes to `SET NULL` and keeps the IP address and user agent with no owner.
   - `pending_data_deletion_request.user_id` has no ON DELETE rule, so a hard delete of a user with a request row will fail.
   - `applied_opportunity.influencer_id` is `NOT NULL ... ON DELETE SET NULL`, so `FirebaseAuthProxyService.deleteAccount` fails for any influencer with applications. It also never cleans Firestore, TOTP or storage.
5. **The Meta paths have gaps:**
   - The deferred path sets `TO_BE_DELETED` without bumping the token version.
   - The deauthorize path catches an exception from `archiveUser` inside its own transaction, which probably rolls back the disconnect too.
   - Firestore data is deleted before the database commits.
   - The status URL sent to Meta, `/deletion-status?code=`, has no matching route in the frontend.
   - When no connection matches, Meta gets a random code that nothing stores, so its status lookup can never succeed.
6. **Logs keep personal data.** The no-consent cron logs `email={}` in clear, and the eligibility summary embeds the email.
7. **Other gaps:** `OrphanCleanupTask` has no ShedLock while `DeferredDeletionCronJob` does, and both run at 03:00. The frontend says the delete endpoint "does not re-guard" eligibility. It does for influencers, but only by throwing a bare `IllegalArgumentException`.

## Proposed change
**One erasure pipeline for every entry point.** Every entry point creates one persisted erasure request. One executor carries it out in steps that are safe to repeat, and the steps are tracked per system.

**Keep the user row as an anonymised stub instead of hard-deleting it.** The alternative, hard delete, would mean rewriting every foreign key and copying out what must be kept. The stub wins because the records we must keep (consent records, invoices, subscription events, application status history) already point at `user.id`, several of them with CASCADE. Emptying the row keeps those links valid and stops CASCADE destroying them. Every personal field becomes null or a fixed placeholder (for example `deleted-{id}@erased.invalid`).

**The lifecycle:**
1. **Request, the same for every source** (user, Meta data-deletion, Meta deauthorize, admin, no-consent cron):
   - Re-check eligibility on the server and return a translatable 409 with the blockers, instead of the `IllegalArgumentException`.
   - Set `TO_BE_DELETED`, bump the token version, evict the cache, and cut off the Firebase session.
   - Disconnect social connections and delete the Firestore Instagram tokens straight away.
   - Write a `ConsentRecord` with the existing source `ACCOUNT_DELETION`.
   - Store the request as `BLOCKED` or `SCHEDULED` (with a grace period) plus a confirmation code. Every request gets a real, stored code, including Meta requests with no matching connection.
2. **Execute**, once the grace period has passed and the blockers are cleared, or immediately for an admin:
   - **Database, in one transaction:**
     - Anonymise the user row.
     - Delete addresses, social connections, preferences, notifications, company data, current-consent rows, file-upload rows and non-retained applications.
     - Anonymise support tickets (contact email, IP address, free text).
     - Truncate IP addresses on retained consent rows.
     - Remove retained applications' Firebase-UID columns, if they exist.
   - **External systems, after the commit, tracked as a `CascadeDeleteTask`:**
     - Delete Firestore Instagram and TOTP data.
     - Delete storage: `users/{uid}/`, `content/{uid}/`, every path in `file_uploads`, and the profile picture.
     - Delete the Firebase Auth user and evict the cache.
     - `OrphanCleanupTask` retries failures, now under a ShedLock.
     - The Firebase UID stays only in the task row until the last external step succeeds, then it is cleared.
3. **Retain, then purge.** A retention job deletes the records we kept once their legal period ends. The periods are settings, and the DPO or legal team must set them. Examples: consent proof for the limitation period, invoices for 5 years after the end of the fiscal year under Polish accounting law.

## Plan
1. **Database migration, additive only:** extend `pending_data_deletion_request` with `source`, `scheduled_for`, `executed_at` and a `cascade_delete_task_id` reference. Update `PendingDataDeletionRequest.java` and `DeletionRequestStatus.java` (add `BLOCKED`, `SCHEDULED`, `EXECUTING`). Nothing changes behaviour yet.
2. **Fix the storage leak now:** `FirebaseStorageService.deleteUserDirectory` also deletes `content/{uid}/` (the prefix taken from `file-upload.storage-path-pattern`) and the paths in `FileUploadRepository.findByUserId`. Touches `storage/service/FirebaseStorageService.java` and `AdminCascadeDeleteServiceImpl.java` (the preview count too).
3. **New `AccountErasureService.requestErasure(user, source)`** in the `user/` package, built from the logic that is in `UserAccountOrchestrator` today. Route these callers to it:
   - `UserService.delete` (self-service)
   - `InstagramDataDeletionService` (both cases; always store the code)
   - `InstagramDeauthorizationService.processInactiveUser` (drop the swallowed exception)
   - `LegalConsentService.archiveUsersWithNoConsents`

   Mask the email in `LegalConsentService.java:547` and in `generateDeletionSummary`.
4. **New `AccountErasureExecutor.execute(requestId)`:**
   - Database transaction as described above. Touches `User.java`, the address, social-connection, preferences, notification, support-ticket, consent and file-upload repositories.
   - External steps registered after commit through `CascadeDeleteTask`.
   - Add `@SchedulerLock` to `OrphanCleanupTask.java`.
5. **Generalise `DeferredDeletionCronJob`:** it executes `SCHEDULED` requests whose time has come and re-checks `BLOCKED` ones. The Meta and in-app sources now share this job.
6. **Route the hard deletes to the executor, skipping the grace period:** `AdminCascadeDeleteServiceImpl.forceDeleteUser` (remove `userRepository.delete(user)` for users), `UserService.deletePermanently` and `FirebaseAuthProxyService.deleteAccount`. The cascade for an opportunity or an application stays as it is.
7. **Database migration, run only after step 6:** change the `invoice_record`, `subscription_event`, `billing_period`, `user_consent` and `consent_record` foreign keys to `RESTRICT`, and give `pending_data_deletion_request` an explicit `RESTRICT`. Any stray hard delete then fails loudly instead of silently destroying retained records.
8. **Retention purge job:** a new cron plus a properties class for the periods, with values supplied by the DPO.
9. **Frontend:**
   - A public `/deletion-status` route and component that calls the generated `instagram-callback-controller` status endpoint.
   - `profile-view.component.ts` handles the server's 409 blockers and shows "scheduled for {date}".
   - `admin-cascade-delete-dialog.component.*` lists what is erased and what is kept.
   - Regenerate the client with `openapi:gen` after steps 3–6.
10. **Tests:** see the next section.

## Risks and invariants
- **A second request must not create a second job, and an interrupted run must be able to resume.** Each user gets one erasure request, and every step checks its status first. Tests: extend `InstagramDataDeletionServiceUnitTest` and `DeferredDeletionCronJobUnitTest`, and add a test that runs the executor twice.
- **No personal data remains after execution.** Add a new `AccountErasureCompletenessIntegrationTest`:
  - Build the table list from `information_schema` (every foreign key to `"user"` plus every `user_id` or Firebase-UID column), so a new table fails the test until someone classifies it as erased or retained.
  - Seed one row in each table.
  - Execute the erasure and assert that no seeded email, name, phone or UID remains.
- **Records we must keep survive:** consent records, invoices and subscription events still point at the anonymised row after execution, and a hard delete is rejected by `RESTRICT`. Tests: extend `UserService_Delete_IntegrationTest` and the `UserAccountOrchestrator_*` suites.
- **A user with active collaborations is not erased until they end:** `UserAccountOrchestrator_Blockers_IntegrationTest` plus a new test for the in-app 409.
- **Sessions die at request time, on every path:** assert that the token version is bumped and the cache evicted on the deferred path too.
- **Storage:** a test with files under both `users/` and `content/` asserts both are deleted.
- **Failure modes to watch:**
  - Unique constraints on email: use the placeholder `deleted-{id}@erased.invalid`.
  - Lookups that key on Firebase UID, such as `appliedOpportunityService.findByInfluencerUserId`, will stop finding retained rows once the UID is cleared.
  - Reactivating a `TO_BE_DELETED` account (`UserService.java:640`) must be refused once execution has started.
  - Meta requires HTTP 200 and a status URL that resolves. Cover it with a contract test on `/deletion-status`.
- **Moving the existing `TO_BE_DELETED` users:** a one-off backfill creates `SCHEDULED` requests for them. Run it after steps 1–5, with a dry-run report first.

## Evidence
- FACT: For an influencer, `archiveUser` sets `TO_BE_DELETED`, bumps the token version, saves and evicts the cache. Its only anonymisation is clearing the two email-verification timestamps; only companies get name and phone replaced. Active applications make it throw `IllegalArgumentException`, and `APPLIED` ones are deleted (`backend/src/main/java/com/sm/instagram/platform/user/UserAccountOrchestrator.java`, read in full).
- FACT: The common and permanent-delete blocker checks are empty placeholders (same file, lines 221–240).
- FACT: A search of `backend/src/main/java` finds no code that sets `AccountStatus.DELETED`. `UserService.java:640` allows reactivation from `TO_BE_DELETED`.
- FACT: `UserService.delete` calls `archiveUser`. `deletePermanently` evicts the cache, runs `repository.delete` and then `firebaseService.deleteUserGraceful` (`UserService.java:340–390`).
- FACT: `FirebaseAuthProxyService.deleteAccount` checks the password, deletes the user row, then `firebaseAuth.deleteUser`, with no Firestore, TOTP or storage cleanup (`FirebaseAuthProxyService.java:805–884`).
- FACT: The frontend calls `checkMyDeletionEligibility`, then `DELETE /users/{id}`, then signs out, and states the endpoint "does not re-guard" (`frontend/src/app/core/user/user.service.ts:67–85`, `frontend/src/app/feature/profile/profile-view.component.ts:199–261`).
- FACT: The data-deletion service marks the connection `DISCONNECTED` and, when nothing blocks, calls `archiveUser` and saves a `COMPLETED` request. Otherwise it sets `TO_BE_DELETED` and evicts the cache with no token-version bump, then queues a `PENDING` request. Firestore deletion runs inside the `@Transactional` method. With no matching connection it returns a random, unstored UUID. The status URL is `frontendUrl + "/deletion-status?code="` (`InstagramDataDeletionService.java`, read in full).
- FACT: The deauthorize service catches exceptions from `archiveUser` inside a `@Transactional` method; for a verified user it only disconnects, deletes Firestore data and sends an email (`InstagramDeauthorizationService.java`, read in full).
- INFERENCE: The deauthorize service's catch leaves the transaction marked rollback-only, because `archiveUser` is a `@Transactional` call on an injected proxy. The disconnect is therefore rolled back as well.
- FACT: `DeferredDeletionCronJob` handles only `PENDING` requests (which only the Meta path creates), re-checks eligibility and calls `archiveUser` under ShedLock (`DeferredDeletionCronJob.java`, read in full).
- FACT: `forceDeleteUser` deletes applications and history, then `userRepository.delete(user)`. Its external cleanup (Firestore Instagram and TOTP, `deleteUserDirectory`, Firebase Auth) runs inside the same `@Transactional` method, tracked by `CascadeDeleteTask` (`AdminCascadeDeleteServiceImpl.java:244–418`).
- FACT: `OrphanCleanupTask` runs at `0 0 3 * * *` with no `@SchedulerLock` (`OrphanCleanupTask.java`, read in full).
- FACT: `deleteUserDirectory` deletes the prefix `users/{uid}/` (`FirebaseStorageService.java:42–53`). Uploads are written to `content/{userId}/{timestamp}_{filename}` (`SignedUrlService.java:65,313–321`; `application.yml:537`, `application-prod.yml:230`). `file_uploads.user_id` is the Firebase UID (`005-file-uploads-table.sql:37`).
- FACT: The foreign-key rules on `user`:
  - `user_consent` and `user_current_consent`: CASCADE (`006-consent-management-tables.sql:41,67`)
  - `consent_record`: SET NULL, and the table stores IP address and user agent (`09-03-2026-consent-module-tables.sql:36`; `ConsentRecord.java:54,57`)
  - `subscription_event`, `billing_period`, `invoice_record`: CASCADE (`22-03-2026-subscription-tables.sql:80,98,114`)
  - `pending_data_deletion_request`: no ON DELETE rule (`11-03-2026-create-pending-data-deletion-request.sql:17`)
  - `applied_opportunity.influencer_id`: `NOT NULL ... ON DELETE SET NULL` (`002-tables.sql:187`)
- INFERENCE: With PostgreSQL's default NO ACTION, hard-deleting a user who has a pending-request row fails. Setting a `NOT NULL` column to null also fails, so `FirebaseAuthProxyService.deleteAccount` breaks for any influencer with applications.
- FACT: `SupportTicket` stores `contactEmail`, `description` and `ipAddress`; `Notification` and `InvoiceRecord` reference `user_id` (the entity grep). `User` holds `email`, `firstName`, `lastName`, `profilePicture`, `phoneNumber`, `nip` and `lastVerifiedEmail` (the `User.java` grep).
- FACT: The no-consent cron logs `email={}` with `user.getEmail()` (`LegalConsentService.java:547–549`). The eligibility summary embeds the email (`UserAccountOrchestrator.java:293`).
- FACT: The only occurrences of `deletion-status` in `frontend/src` are in the generated client (`frontend/src/app/api/api/instagram-callback-controller.api.ts`); no route exists.
- FACT: The `ConsentSource` enum already has `ACCOUNT_DELETION` (`legal/ConsentSource.java:10`).
- FACT: Both the in-memory and Redis user caches treat `TO_BE_DELETED` as unable to authenticate (`InMemoryUserCache.java:52–56`, `RedisUserCache.java:45–50`).
- INFERENCE: The graph shows `UserAccountOrchestrator` used by `UserService`, `LegalConsentService`, `InstagramDataDeletionService`, `InstagramDeauthorizationService` and `DeferredDeletionCronJob`, so those five are the full set of callers to reroute in step 3. Absent edges are unknown, and step 3 should confirm with a search.
- HYPOTHESIS: `applied_opportunity` has a Firebase-UID column, because the dev seed inserts `u.firebase_user_id` and `findByInfluencerUserId` takes a Firebase UID. Check `AppliedOpportunity.java` before step 4.
- HYPOTHESIS: `FirebaseService` can revoke refresh tokens or disable a user. Search it before step 3; otherwise add a thin wrapper over the Admin SDK.
- HYPOTHESIS: `FirebaseAuthProxyController.java:587` logs the unmasked email; the format string has `Email={}` but I did not read the arguments.
- HYPOTHESIS: Firestore holds user-cache or audit documents beyond `instagramUsers` and `totpSecrets` (the TOTP delete log mentions `audit_logs`). Read `FirestoreService` and `TotpFirestoreService` before step 4.
- Open, not answerable from the code: the retention periods and which consent and application fields must be kept. The DPO or legal counsel must set them before step 8.
