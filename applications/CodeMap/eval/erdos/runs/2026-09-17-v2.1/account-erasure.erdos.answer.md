# Account erasure in checkItOut: where an influencer's personal data survives, and a design that guarantees deletion while keeping legally required records

## Problem

An influencer asks for their account to be erased. Today there are five ways into deletion:
- the self-service "delete account" button;
- the Meta data-deletion callback;
- the Meta deauthorization callback;
- the admin "delete permanently" action;
- the admin cascade delete.

None of them removes the person's data from every store, and one of them (self-service) never gets past a soft-delete flag. The admin paths also get the balance wrong the other way: some records the law says to keep (invoices, consent history) are deleted along with the user row.

"Done" means three things:
- every request, whatever its source, ends in a verified, retryable erasure of Postgres, Firebase Auth, Firestore, Storage and the caches;
- only the records the law requires survive, stripped of personal data and with a retention date;
- the schema can't quietly gain a new table holding personal data that erasure doesn't know about.

## Where it lives today

**Backend modules and files**
- **Eligibility and soft delete.** `backend/src/main/java/com/sm/instagram/platform/user/UserAccountOrchestrator.java` has `archiveUser` and `checkDeletionEligibilityForUser`. The blocker DTOs are in `user/dto/DeletionBlocker*.java`.
- **Entry points.**
  - `user/UserController.java`: `DELETE /users/{ids}`, `DELETE /users/delete-permanently/{ids}`, `GET /users/me/deletion-eligibility`.
  - `user/UserService.java`: `delete` at line 342 and `deletePermanently` at line 357.
- **Deferred deletion.** `auth/cron/DeferredDeletionCronJob.java`, `auth/entity/PendingDataDeletionRequest.java` and `auth/repository/PendingDataDeletionRequestRepository.java`.
- **Meta callbacks.** `auth/controller/InstagramCallbackController.java` (`/auth/instagram/deauthorize`, `/data-deletion`, `/deletion-status`), backed by `auth/service/InstagramDataDeletionService.java`, `auth/service/InstagramDeauthorizationService.java` and `auth/service/MetaSignedRequestService.java`.
- **Admin cascade.** `admin/cascade/AdminCascadeDeleteController.java`, `AdminCascadeDeleteServiceImpl.java` (1009 lines), `CascadeDeleteTask.java` (tracks the status of each store) and `OrphanCleanupTask.java` (retries at 03:00).
- **External stores.**
  - `storage/service/FirebaseStorageService.java`
  - `auth/firebase/FirestoreService.java` (`deleteInstagramUserData`)
  - `auth/firebase/TotpFirestoreService.java` (`deleteTotpData`)
  - `auth/firebase/FirebaseService.java` (`deleteUserGraceful`)
  - `auth/cache/UserCacheService.java`, with `RedisUserCache` and `InMemoryUserCache`
- **Consent.**
  - `legal/ConsentRecord.java` and `legal/LegalConsentService.java` (its no-consent cleanup also calls `archiveUser`)
  - `consent/UserConsent.java` and `consent/UserCurrentConsent.java`
- **Schema (Liquibase).**
  - `db/changelog/common/001-schema/002-tables.sql`
  - `006-consent-management-tables.sql`
  - `2026/01/2026-01-create-notifications-table.sql`
  - `2026/03/09-03-2026-consent-module-tables.sql`
  - `2026/03/11-03-2026-create-pending-data-deletion-request.sql`
  - `2026/03/11-03-2026-create-company-data-table.sql`
  - `2026/03/22-03-2026-subscription-tables.sql`

**Frontend**
- `frontend/src/app/feature/profile/profile-view.component.ts` (`deleteAccount`): checks eligibility, then shows the blockers dialog or the confirmation dialog, then calls `DELETE /users/{id}` and signs out.
- `frontend/src/app/core/user/user.service.ts`.
- Admin side: `frontend/src/app/core/admin/cascade-delete.service.ts` and `feature/opportunities/admin-cascade-delete-dialog.component.ts`.

### Flows as they run today

1. **Self-service delete.**
   - The profile page calls `DELETE /users/{id}`, which runs `UserService.delete` and then `archiveUser`.
   - For an influencer, `archiveUser` deletes pending (APPLIED) applications, clears two email-verification timestamps, sets `TO_BE_DELETED`, bumps the token version, saves the user and clears the user cache.
   - **Nothing else happens, ever.** No deletion request is queued, and the cron job only reads `PENDING` requests.
   - The frontend comment claiming "DeferredDeletionCronJob performs the RODO cascade later" is wrong.
2. **Meta data-deletion callback.**
   - Marks the Instagram connection `DISCONNECTED` and checks eligibility.
   - **(a) No blockers:** runs `archiveUser` and records a `COMPLETED` request.
   - **(b) Blockers:** sets `TO_BE_DELETED` without calling `archiveUser`, and records a `PENDING` request.
   - Either way it then deletes the Instagram data in Firestore, and a failure there is only logged.
3. **Deferred cron** (03:00, with a distributed lock): for each `PENDING` request, re-checks eligibility, runs `archiveUser` and marks the request `COMPLETED`. This is still only a soft delete.
4. **Meta deauthorization.**
   - If the email is verified: disconnect Instagram, delete the Firestore Instagram data, send an email.
   - Otherwise: `archiveUser` (inside a try/catch) plus the Firestore delete.
5. **Admin "delete permanently".**
   - Clears the cache, deletes the user row, flushes, then deletes the Firebase Auth account, all inside one transaction.
   - Firestore, TOTP and Storage are never touched, and nothing is tracked or retried.
6. **Admin cascade delete** (`forceDeleteUser`):
   - Deletes applications and their history, partnership opportunities and the user row (addresses and social connections cascade).
   - Then, still inside the same transaction, deletes Firestore Instagram data, the TOTP secret, Storage `users/{uid}/` and Firebase Auth, and clears the cache.
   - Records a `CascadeDeleteTask`, which `OrphanCleanupTask` retries up to 3 times.

### Where personal data survives, or legally required records are lost

| # | Gap | Evidence |
|---|---|---|
| G1 | A self-service delete stays `TO_BE_DELETED` forever. The user row keeps email, name, phone and Firebase UID. Addresses, preferences, social connections, notifications, the Firebase Auth account, Firestore Instagram data, the TOTP secret and stored files all remain. | FACT |
| G2 | `archiveUser` removes no identity data for an influencer. Only companies get first/last name replaced with "N/A" and phone cleared, and even then the email stays. The deauthorization comment "sets TO_BE_DELETED, anonymizes PII" is false for influencers. | FACT |
| G3 | The Instagram connection is only marked `DISCONNECTED`. It keeps `socialUserId`, `email`, `profilePictureUrl` and `followersCount`. | FACT |
| G4 | Storage clean-up deletes only `users/{uid}/`, but user files also live under `content/{uid}/` (`FileManagementController` allows both prefixes). Support-ticket attachments are stored as `fileUrl`, with an unknown prefix. | FACT / HYPOTHESIS |
| G5 | `deletePermanently` will fail for most influencers. `pending_data_deletion_request.user_id` has no `ON DELETE` action, so it blocks. `applied_opportunity.influencer_id` is `NOT NULL` yet `ON DELETE SET NULL`, which is a contradiction. It also skips Firestore, TOTP and Storage, and deletes Firebase Auth before the commit (its own comment says "AFTER PG commit"). | FACT (schema + code); INFERENCE (runtime failure) |
| G6 | `forceDeleteUser` deletes external data before Postgres commits. A foreign-key error raised at flush or commit (for example from `pending_data_deletion_request`) would roll back the Postgres delete and the task row after Firebase Auth and Storage are already gone. That leaves a half-erased user with no retry record. It also never deletes `pending_data_deletion_request`. | INFERENCE |
| G7 | Some rows survive a hard delete because their user link is `ON DELETE SET NULL`:<br>• `consent_record` keeps `ip_address` and `user_agent`<br>• `support_ticket` keeps `contact_email`, `description` and `ip_address`, plus its responses and attachments<br>• `search_query_log` keeps `ip_address`<br>• `applied_opportunity_status_history` keeps `changed_by_firebase_id` | FACT |
| G8 | Counterparties' notifications keep `influencer_id`, a `snapshot` JSONB field and a `message` that probably names the influencer. Only the recipient's own rows cascade. | FACT (columns); HYPOTHESIS (snapshot content) |
| G9 | A hard delete of a company also deletes `invoice_record`, `billing_period`, `subscription_event` and `company_data` (all `ON DELETE CASCADE`). That breaks accounting retention. `user_consent` also cascades, which destroys evidence of consent. | FACT (schema); retention rule is legal INFERENCE |
| G10 | Email addresses end up in logs and the admin summary: `LegalConsentService` logs `email=` when archiving, and the eligibility summary and DTO carry the email. | FACT |
| G11 | Meta's status link points to `frontendUrl + "/deletion-status"`, and I found no frontend route for it. | HYPOTHESIS |
| G12 | Deauthorization for an unverified user swallows `archiveUser` failures inside `@Transactional`. `OrphanCleanupTask` has no distributed lock and fires at the same time as the deferred cron. | FACT / INFERENCE |

## Proposed change

**Design: one erasure pipeline, one request record, one tracked task.** Every entry point creates or reuses a `PendingDataDeletionRequest`. One executor then performs erasure in two phases, reusing mechanisms the project already has:
- the deferred cron job with its distributed lock;
- `CascadeDeleteTask` for per-store status and retries;
- `UserCacheService.evict` and the token-version bump to kill sessions.

1. **Request (synchronous, all five entry points).**
   - Check eligibility on the server as well; the DELETE endpoint currently relies on the client.
   - Set `TO_BE_DELETED`, bump the token version, clear the cache.
   - Create a request with `requestedAt`, `source` (SELF / META_DELETION / META_DEAUTH / ADMIN / NO_CONSENT), blockers and `eraseAfter` (a short grace period for self-service requests; immediate for Meta and admin).
   - This is where today's `archiveUser` fits: it becomes "suspend", not "delete".
2. **Execute (`DeferredDeletionCronJob` → new `AccountErasureService`).** When the request is unblocked and due, run phase A and then phase B.
   - **Phase A (one Postgres transaction).** Apply a retention matrix in which each module registers an eraser for its tables:
     - delete addresses, preferences, social connections, own notifications, `user_consent`/`user_current_consent`, and draft or unfinished applications;
     - **anonymise** what must or should be kept:
       - `consent_record`: keep `document_id`, `timestamp`, `source`; drop or truncate IP and user agent (legal to decide), set a retention date;
       - support tickets: replace email with a placeholder, blank description and IP, delete attachments;
       - `search_query_log`: blank the IP;
       - status history: blank `changed_by_firebase_id`;
       - counterparties' notifications: scrub snapshot and message;
       - completed collaborations that a company needs as business records: keep them, pointing at the tombstone, content files deleted;
     - turn the `user` row into a **tombstone**: status `DELETED`, email/name/phone/avatar blanked, `firebase_user_id` replaced by `erased:<uuid>`, `erasedAt` set;
     - record the external steps as `PENDING` on a `CascadeDeleteTask` in the same transaction.
   - **Phase B (after commit).** Delete Firestore Instagram data, the TOTP secret, Storage `users/{uid}/`, Storage `content/{uid}/` and ticket attachments, and Firebase Auth last. Clear the cache. Update the task; `OrphanCleanupTask` retries whatever failed.

**Why a tombstone instead of a hard delete.**
- The rows the law requires (invoices, billing periods, subscription events and company data for companies; consent proof for everyone) hang off `NOT NULL` foreign keys to `user`. A PII-free tombstone keeps them valid without rewriting those constraints.
- It stops the `ON DELETE CASCADE` rules from destroying records that must be retained (G9).
- It keeps Meta's confirmation-code lookup working.

The alternative is a true hard delete with every retained table's link changed to `SET NULL` and its own copy of the needed data. That is cleaner for the influencer-only case but needs a migration per table and loses the link for audits. I recommend the tombstone.

The admin cascade and "delete permanently" become thin callers of the same executor with `eraseAfter = now`. `forceDeleteUser` keeps its preview.

**Completeness guard.** An integration test reads `information_schema` for every foreign key to `"user"` and every column named like `*email*`, `*ip*`, `firebase*` or `*user_agent*`. It fails when a column is not covered by a registered eraser or an explicit "retained" entry in the matrix. This is what makes future schema changes safe.

Legal counsel must confirm the retention matrix (IP on consent records, how long collaboration records are kept, invoice retention under Polish accounting law). The code should carry the matrix as data, so that decision doesn't need a redesign.

## Plan

1. **Tests that describe today's gaps (no behaviour change).**
   - Add a `useraccountorchestrator` integration test that seeds an influencer with rows in every table linked to `user` plus fake Firestore/Storage objects, runs today's flows and records what survives.
   - Add the `information_schema` coverage test, marked expected-to-fail.
   - Files: `backend/src/test/java/com/sm/instagram/platform/integration/service/useraccountorchestrator/*` (new), `UserAccountOrchestratorIntegrationTestBase.java`.
2. **Schema migration (additive, safe to deploy alone).** New changeset under `db/changelog/2026/09/`:
   - `pending_data_deletion_request`: add `source`, `erase_after`, `cascade_task_id`; change its user FK to `ON DELETE CASCADE` so tombstones and legacy hard deletes stop being blocked.
   - `user`: add `erased_at`.
   - `consent_record`: add `retain_until`.
   - `cascade_delete_task`: add the missing store columns (content storage, attachments) or a generic `step_statuses JSONB`.
   - `applied_opportunity.influencer_id`: fix `NOT NULL` + `SET NULL` to `ON DELETE RESTRICT`, so the tombstone is the only path.
   - `invoice_record`, `billing_period`, `subscription_event`, `company_data`: change the user FK to `ON DELETE RESTRICT` so a hard delete can never wipe accounting records.
3. **Eraser interface and per-module implementations.**
   - New `user/erasure/UserDataEraser.java` (method: erase the user's rows within the transaction, return the tables and counts).
   - Implementations in: `address/`, `userpreferences/`, `usersocialconnection/`, `notification/` (own rows plus counterparty snapshots), `consent/` + `legal/` (delete `user_consent*`, anonymise `consent_record`), `support/ticket/services/`, `appliedopportunities/` (delete open items, keep completed ones on the tombstone), search log, `subscription/` (company: keep and anonymise contact data only).
   - New `user/erasure/RetentionMatrix.java` records what is retained and why.
4. **`AccountErasureService`** (new, `user/erasure/`).
   - Phase A: all erasers plus the tombstone in one `@Transactional`, and a `CascadeDeleteTask` with the external steps pending.
   - Phase B: a `TransactionSynchronization.afterCommit` or `@TransactionalEventListener(AFTER_COMMIT)` handler, following the pattern the notifications module already uses. It calls `FirestoreService.deleteInstagramUserData`, `TotpFirestoreService.deleteTotpData`, `FirebaseStorageService` and `FirebaseService.deleteUserGraceful` (last), then `UserCacheService.evict`.
   - Extend `FirebaseStorageService` with `deleteUserContent(uid)` for `content/{uid}/` and a method to delete ticket attachments by URL.
5. **Request step unified.**
   - Rename `UserAccountOrchestrator.archiveUser` to `suspendForDeletion`: it keeps the status, token-bump and cache logic, stops half-anonymising, and creates the `PendingDataDeletionRequest`.
   - Callers to update: `UserService.delete`, `InstagramDataDeletionService` (both branches, which also adds the token bump to the deferred branch), `InstagramDeauthorizationService` (unverified branch; let failures propagate instead of swallowing them in the transaction), and `LegalConsentService`'s no-consent cleanup.
   - Put a server-side eligibility check in `UserService.delete`.
6. **Executor wired to the crons.**
   - `DeferredDeletionCronJob.processOneRequest` calls `AccountErasureService.erase` when the request is unblocked and `erase_after` has passed, then marks it `COMPLETED` with `cascade_task_id`.
   - `OrphanCleanupTask`: add `@SchedulerLock`, move it off 03:00, and have `retryTask` cover the new steps.
7. **Admin paths delegate.**
   - `AdminCascadeDeleteServiceImpl.forceDeleteUser` and `UserService.deletePermanently` create a request with source ADMIN and run the executor immediately.
   - Update `previewUserDeletion` to list what is retained versus erased, from `RetentionMatrix`.
   - Remove the pre-commit Firebase calls.
8. **Personal data in logs and DTOs.**
   - Replace `email=` in `LegalConsentService`'s archive log and in `generateDeletionSummary` with a masked value (`PiiMaskingUtils` exists in `common`).
   - Don't include email in `DeletionEligibilityDto` for the cron and Meta paths.
9. **Frontend.**
   - `frontend/src/app/core/user/user.service.ts`: fix the doc comment.
   - `feature/profile/profile-view.component.ts` + `delete-confirmation-dialog.component.*`: say "your account will be erased on <date>; these records are kept for legal reasons", using a new `eraseAfter` field on the response. Regenerate the API client with `openapi:gen`.
   - Add a public `/deletion-status` route calling `GET /auth/instagram/deletion-status` (after checking G11).
   - Admin `cascade-delete.service.ts` and the dialog show the retained/erased breakdown.
10. **Backfill (one-off, run after deploy).** Create `PENDING` requests (source BACKFILL) for existing users stuck in `TO_BE_DELETED`, so the cron erases them. Do a dry run through the preview first.

## Risks and invariants

- **Every erasure is recorded.** Each erasure leaves one `COMPLETED` request and one `CascadeDeleteTask` with no `FAILED` step after retries. Stuck tasks keep raising the existing `ALERT` log.
  - Guards: the step 1 test (now expected to pass), `DeferredDeletionCronJobUnitTest`, an `OrphanCleanupTask` test.
- **External data is never deleted while Postgres can still roll back.**
  - Guard: an integration test that forces a failure in phase A and asserts Firebase and Storage mocks were never called.
- **Idempotency.** A second Meta callback, a second cron run, or an admin re-run on a tombstone does nothing and returns the existing confirmation code.
  - Guards: `InstagramDataDeletionServiceUnitTest`, `InstagramDeauthorizationServiceUnitTest`, a new "erase twice" test.
- **Retention.** Invoices, billing periods, subscription events and company data survive for companies, and consent records survive with `retain_until`. The `RESTRICT` foreign keys make any hard-delete regression fail loudly.
  - Guard: a company erasure test asserting invoice rows exist with no PII left in the user row.
- **Completeness.** No column in the covered list still holds the seeded marker values, in Postgres or in the Firestore/Storage fakes. The `information_schema` guard fails on any new foreign key to `user` that isn't covered.
- **Authorization and sessions.** The token bump and cache eviction happen when the request is made, not at execution time. Otherwise a stale Redis entry lets the user keep acting for up to the 5-minute TTL mentioned in `InstagramDataDeletionService`.
- **Eligibility.** Active collaborations still defer erasure. The server now enforces this on `DELETE /users/{id}` too, so the frontend dialog is no longer the only gate.
  - Guards: `UserAccountOrchestrator_Blockers_IntegrationTest`, `UserAccountOrchestrator_DeletionEligibility_IntegrationTest`.
- **Last admin.** Keep the check, now in both the orchestrator and the executor.
- **Migration risk.**
  - Before the `applied_opportunity` FK change, confirm no rows have a null `influencer_id`.
  - Before switching to `RESTRICT`, confirm no scheduled job hard-deletes users.
  - The backfill should run with rate limits (Firebase API quotas).
- **Meta contract.** The callbacks must still answer 200 with a `url` and `confirmation_code`, and the status URL must resolve.
- **Unknowns.** Stripe and Fakturownia customer records, and the contents of the notification `snapshot` field, weren't read. Ticket attachment storage paths weren't confirmed.

## Evidence

**FACT (lines read)**
- `archiveUser` for INFLUENCER only deletes APPLIED applications and clears `emailVerificationSentAt`/`emailVerifiedAt`. COMPANY sets first/last name to "N/A" and phone to null. Neither clears email, Firestore, Storage or Firebase Auth. (`backend/.../user/UserAccountOrchestrator.java`, lines 55–87 and 244–281)
- `checkCommonSoftDeleteBlockers` and `checkPermanentDeleteSpecificBlockers` are placeholders. (`UserAccountOrchestrator.java`, lines 221–240)
- `UserService.delete` only calls `archiveUser`. `deletePermanently` evicts the cache, deletes and flushes, then calls `firebaseService.deleteUserGraceful` inside the same `@Transactional` method. (`backend/.../user/UserService.java`, lines 340–390)
- `DeferredDeletionCronJob` processes only `findAllByStatus(PENDING)` and calls `archiveUser`. (`backend/.../auth/cron/DeferredDeletionCronJob.java`, lines 56–109)
- The Meta data-deletion flow: deferred branch sets `TO_BE_DELETED` and evicts the cache without a token bump; the Firestore delete's failures are logged only; the status URL is `frontendUrl + "/deletion-status"`. (`backend/.../auth/service/InstagramDataDeletionService.java`, lines 103–186)
- Deauthorization for an unverified user wraps `archiveUser` in try/catch inside `@Transactional processDeauthorization`. (`InstagramDeauthorizationService.java`, lines 43–126)
- The callbacks always return 200, and the backend status endpoint is `/auth/instagram/deletion-status`. (`InstagramCallbackController.java`)
- `forceDeleteUser` is `@Transactional`: Postgres delete, then Firestore/TOTP/Storage/Auth deletes, then cache eviction and task save, all in the same method. The cascade does not touch `pending_data_deletion_request`. (`AdminCascadeDeleteServiceImpl.java`, lines 243–418)
- `OrphanCleanupTask` has `@Scheduled(cron = "0 0 3 * * *")` and no `@SchedulerLock`. (`OrphanCleanupTask.java`, line 38)
- `deleteUserDirectory` deletes only `"users/" + firebaseUid + "/"`. (`FirebaseStorageService.java`, lines 42–52)
- Ownership checks accept `users/{uid}/` and `content/{uid}/`. (`FileManagementController.java`, lines 318–320 and 370–371)
- Schema, `ON DELETE` actions:
  - `address`, `user_social_connection`, `user_preferences`: CASCADE
  - `applied_opportunity.influencer_id`: `NOT NULL` + `SET NULL`
  - `status_history.changed_by_user_id`, `support_ticket.user_id`, `search_query_log.user_id`: SET NULL (all in `002-tables.sql`)
  - `user_consent` and `user_current_consent`: CASCADE (`006-consent-management-tables.sql`)
  - `consent_record`: SET NULL (`09-03-2026-consent-module-tables.sql`)
  - `pending_data_deletion_request`: no action (`11-03-2026-create-pending-data-deletion-request.sql`)
  - `company_subscription`, `subscription_event`, `billing_period`, `invoice_record`: CASCADE (`22-03-2026-subscription-tables.sql`)
  - `company_data`: CASCADE (`11-03-2026-create-company-data-table.sql`)
  - `notifications.user_id`: CASCADE, with non-FK `influencer_id`, `snapshot` JSONB and `message` columns (`2026-01-create-notifications-table.sql`)
- Personal-data columns: `ConsentRecord` and `UserConsent` have `ip_address` and `user_agent`. `SupportTicket` has `contactEmail`, `description`, `ipAddress`. `TicketAttachment` and `ResponseAttachment` have `fileUrl`. `UserSocialConnection` has `socialUserId`, `profilePictureUrl`, `email`, `followersCount`.
- `LegalConsentService` logs `email={}` when archiving a no-consent account, then calls `archiveUser`. (`LegalConsentService.java`, lines 544–550)
- The frontend deletes via `DELETE /users/{id}` after a client-side-only eligibility check, and its comment claims the cron does the cascade. (`frontend/src/app/feature/profile/profile-view.component.ts`, lines 200–259; `frontend/src/app/core/user/user.service.ts`, lines 67–85)

**INFERENCE**
- `deletePermanently` and `forceDeleteUser` fail at flush or commit for any user who has a `pending_data_deletion_request` row, because that foreign key has no delete action and neither method removes the row.
- In `forceDeleteUser`, a Postgres failure at commit happens after the external deletes and rolls back the task row too, so nothing is left to retry.
- A swallowed exception from `archiveUser` (a transactional bean joining the outer transaction) marks the deauthorization transaction rollback-only, so the `DISCONNECTED` change is lost as well.
- Subscriptions and invoices belong to companies only (per the "company accounts only" comment on the `company_subscription` table), so an influencer's legally retained data is mainly consent records and possibly collaboration records.

**HYPOTHESIS**
- No frontend route exists for `/deletion-status`. My grep of `frontend/src/app` (excluding `api/`) found none. Check `app.routes.ts` and the lazy route files.
- The notification `snapshot` and `message` fields contain the influencer's name. Check `NotificationEventListener.java` where the snapshot is built.
- Ticket attachments live outside `users/{uid}/`. Check the upload path in `SupportTicketService.java`.
- Stripe and Fakturownia keep customer records that need anonymising for companies. Check `SubscriptionService.deactivateForAccountDeletion` and the Stripe adapter.
- `AnonymousConsentCleanupCronJob` and `NoConsentAccountCleanupCronJob` behave as their names suggest. Read them before relying on them in the retention matrix.
- The retention periods themselves (consent proof, IP addresses, collaboration records, invoices) are for the company's legal counsel or DPO to settle, not the code.
