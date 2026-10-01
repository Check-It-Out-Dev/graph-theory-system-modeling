I don't recommend a second global switch. `app.payments.enabled` should stay the switch that turns the payment machinery on for the whole deployment, and a new per-company allow-list should decide who can start paying. With that list switched on, the settings in use today behave as they do now. The one exception is a grace-period cron, covered in step 5.

## Problem
Payments are off for the whole deployment today (`APP_PAYMENTS_ENABLED` defaults to `false`). The goal is to let a named set of pilot companies start trials and paid plans, with Stripe and Fakturownia invoicing live. Every other company must see and get exactly what it gets now: the FREE plan only, no paid buttons, no trial offer, and no change in what the scheduled jobs do to it. It is done when the backend and frontend agree on who can pay and the backend enforces it. Tests must pin both the pilot path and the unchanged path for everyone else.

## Where it lives today
Backend, package `com.sm.instagram.platform.subscription`. There is one property, `app.payments.enabled`, read in two ways:

**1. It decides which Spring components exist at startup.** Each of these carries `@ConditionalOnProperty(app.payments.enabled=true)`:
- `.../subscription/SubscriptionPaidController.java`: trial, subscription consent, upgrade, downgrade and its cancel, billing portal, and `/config`. It is also hidden from OpenAPI.
- `.../subscription/stripe/StripeWebhookController.java`: `POST /webhooks/stripe`, with signature checking.
- `.../subscription/stripe/StripeConfig.java`: sets `Stripe.apiKey`.
- `.../subscription/cron/InvoiceRetryCronJob.java` and `.../subscription/cron/TrialExpiryNotifierCronJob.java`.
- `.../subscription/event/InvoiceCreatedEventListener.java`: the immediate Fakturownia send after an invoice is saved.

The opposite case (`=false`) loads `.../subscription/config/PaymentsDisabledBootGuard.java`. It stops startup if any subscription is in a paid or in-progress state: trial, Business, Enterprise, downgrade pending, payment failed, or terms pending.

**2. Code checks the flag while running** (via `AppPaymentsProperties.isEnabled()`):
- `.../subscription/SubscriptionService.java` `requirePaymentsEnabled()` throws `PaymentsDisabledException`. It is called by trial activation, upgrade, downgrade, cancel-downgrade and all five Stripe webhook handlers.
- `.../subscription/cron/SubscriptionPeriodProcessorCronJob.java`: the trial-expiry and downgrade steps only run when the flag is on. The FREE billing-period rollover always runs.
- `.../subscription/cron/TermsGraceProcessorCronJob.java`: does nothing when the flag is off.
- `.../common/publicconfig/PublicConfigController.java`: anonymous `GET /public-config` returns `{paymentsEnabled}`, not cached.

**Always on, regardless of the flag:**
- `SubscriptionController` (status, invoices).
- `CampaignLimitService.enforceLimit`, called by `PartnershipOpportunityService`.
- `SubscriptionService.deactivateForAccountDeletion`, which cancels Stripe objects when their IDs are present.
- `StripeService` (no flag check at all).
- `FakturowniaConfig` (not gated; `fakturownia.enabled: true` in `application.yml`).
- `StripeDevEventPoller`, which only runs under the `dev-poller` profile.

**Configuration and data:**
- `backend/src/main/resources/application.yml:410-411` holds the flag. Lines 628-640 hold the Stripe keys, webhook secret and price IDs.
- Liquibase changeset `db/changelog/2026/04/09-04-2026-downgrade-all-to-free-for-rollout.sql` reset every subscription to FREE and cleared the Stripe IDs.
- A company's subscription is one `company_subscription` row per company user (`user_id` is unique).

**Frontend:**
- `frontend/src/app/core/config/public-config.service.ts` caches `paymentsEnabled` and falls back to `true` if the call fails.
- `frontend/src/app/feature/plan-billing/plan-billing.component.ts:151` and `.html:215,264,275,302,355` show or hide the trial, upgrade, cancel-trial, portal and paid sections based on it.
- `frontend/src/app/layout/layout.component.ts:254-268` shows the trial-offer banner only for COMPANY users, and only if `paymentsEnabled`, then calls `/subscription/status`.
- `frontend/src/app/core/subscription/subscription.service.ts` has separate read and write clients built on `core/api-frozen/subscription.client.ts` and `hidden-models.ts`.
- Fixtures: `sandbox/fixtures/plan-billing.fixture.ts` (stub config with payments on or off) and `core/demo/demo-fixtures.ts:1619`.

**Tests pinned to the switch:**
- Backend unit tests: `SubscriptionService_PaymentsToggleUnitTest`, `PaymentsDisabledBootGuardUnitTest`, `PublicConfigControllerUnitTest`. The other `SubscriptionService_*UnitTest` classes mock the flag to `true`.
- Backend integration tests: `PaymentsTogglePresence_IntegrationTest`, `SubscriptionService_PaymentsDisabled_IntegrationTest`, `PublicConfigController_IntegrationTest`.
- Backend e2e: `RunPaymentsDisabledIT` with `features/payments_off/payments-off.feature`, run in its own Failsafe fork with `-Dapp.payments.enabled=false` (`pom.xml:1913-1935`).
- Test property files set the flag to `true`.
- Frontend: `plan-billing.component.spec.ts`, `layout.component.spec.ts`, `e2e-tests/integration/flows/payments-off.spec.ts`, `e2e-tests/bdd/features/payments-off.feature`.

## Proposed change
Split "the deployment can take payments" from "this company may start paying".

1. **Keep `app.payments.enabled` as the machinery switch.** Set it to `true` for the pilot so the startup-gated Stripe SDK, webhook, invoicing listener and crons exist. They have to exist for anyone to pay, and they already process only rows that have Stripe IDs, invoice records or paid states.
2. **Add an allow-list mode, `app.payments.pilot.enabled`** (default `false`). When it is `false`, every company is allowed, which is exactly today's `enabled=true` behaviour, so existing tests and property files keep their meaning. When it is `true`, only companies in a new `payments_pilot_company(user_id PK/FK, added_at, added_by, note)` table are allowed.
   - Why a database table rather than a list of IDs in configuration: membership changes without a redeploy, leave an audit trail, and can be refused by a service when that would strand a paying company. A config list is simpler, but every pilot change would mean a restart.
3. **Add `PaymentsEligibilityService.isAllowed(userId)`** as the single place that answers the question, in `subscription/config` next to `AppPaymentsProperties`:
   - It returns `enabled && (!pilot.enabled || pilotRepo.existsById(userId))`.
   - `SubscriptionService` uses it where a company *starts* paying: `activateTrial`, `initiateUpgrade`, and `LegalConsentService.recordSubscriptionConsent` via the controller. It reuses `PaymentsDisabledException`.
   - `isTrialEligible` also requires it, so `/status` never offers a trial to a company outside the pilot.
4. **Leave the paths that finish or exit a paid plan on the global check only.** These are the webhook handlers, downgrade, cancel-downgrade, the portal and account-deletion cancellation. If a company is removed from the pilot while it holds a Stripe subscription, renewals, failures and cancellations must still be recorded, and the company must still be able to leave.
5. **Let the frontend learn eligibility per company from the status call it already makes.** Add `paymentsAvailable` to `SubscriptionStatusDtoOut`. `/public-config` stays anonymous and global, because it cannot know the company. The frontend shows paid actions only when `paymentsEnabled && status.paymentsAvailable`, and the server enforces the rule either way.

## Plan
Each step leaves the system working. Steps 1–6 ship with the flag still `false` in production.

1. **Add the pilot table.** New Liquibase changeset under `backend/src/main/resources/db/changelog/2026/<mm>/` in the same formatted-SQL style: `payments_pilot_company`, with an FK to the users table and `ON DELETE CASCADE`. Include it in the master changelog. It starts empty.
2. **Add the allow-list setting.** In `AppPaymentsProperties.java`, add a nested `pilot.enabled` (default `false`). In `application.yml:410`, add `pilot: enabled: ${APP_PAYMENTS_PILOT_ENABLED:false}`.
3. **Add the repository and the eligibility check.** Add `PaymentsPilotCompany` (entity) and `PaymentsPilotCompanyRepository`, plus `PaymentsEligibilityService` in `subscription/config`.
4. **Enforce on the backend.** In `SubscriptionService.java`:
   - Add `requirePaymentsAllowedFor(userId)`, which runs the global check and then the eligibility check, to `activateTrial` (line 61) and `initiateUpgrade` (line 137).
   - Add eligibility to `isTrialEligible` (line 750).
   - Set `paymentsAvailable` in `getStatus` (line 115).
   - Leave the webhook, downgrade and cancel handlers on `requirePaymentsEnabled()`.

   In `SubscriptionPaidController.recordConsent`, check eligibility before `legalConsentService.recordSubscriptionConsent`. Add `paymentsAvailable` to `SubscriptionStatusDtoOut.java`.
5. **Keep the grace cron away from companies outside the pilot.** When the flag is `true`, `TermsGraceProcessorCronJob` starts suspending *every* `TERMS_PENDING` row whose grace period has run out, FREE companies included, and today it does nothing. While the allow-list is on, limit `processExpiredGracePeriods` and `processTrialExpiryInTermsPending` to allowed companies. Either add repository query variants joined to `payments_pilot_company`, or filter in `SubscriptionService.java:856-887` and `624-639`.
6. **Add a startup guard for pilot mode.** New `PaymentsPilotBootGuard` (loaded when `app.payments.pilot.enabled=true`), modelled on `PaymentsDisabledBootGuard`. It counts companies in paid or in-progress states that are not on the allow-list and fails startup with the same kind of message. Add a count query to `CompanySubscriptionRepository.java`.

   Any admin action that removes a company from the pilot should refuse while that company is in one of those states. If there is no admin action at first, membership is managed by audited SQL, and this guard catches mistakes.
7. **Update the frontend.**
   - Add `paymentsAvailable?: boolean` to `frontend/src/app/core/api-frozen/hidden-models.ts` (`SubscriptionStatusDtoOut`) and to `frontend/src/testing/contract/subscription.contract.ts`.
   - In `plan-billing.component.ts`, add a `paidActionsAvailable` computed signal (`paymentsEnabled() && status?.paymentsAvailable !== false`) and replace the `paymentsEnabled()` uses at `.html:215,275,302,355`. Line 264 is the "payments unavailable" notice and should show for the negated value.
   - In `layout.component.ts:259-264`, require `sub.paymentsAvailable` for the trial offer.
   - Update `plan-billing.fixture.ts`, `demo-fixtures.ts` (the `/subscription/status` response) and the two component specs.
   - Decide what a missing field means. Treating it as `true` keeps a newer frontend working against an older backend; since the server enforces the rule anyway, that is safe.
8. **Add tests** (see Risks), including a new Failsafe fork `RunPaymentsPilotIT` in `pom.xml` next to lines 1913-1935. It runs with `-Dapp.payments.enabled=true -Dapp.payments.pilot.enabled=true` and uses a `@payments-pilot` feature.
9. **Roll out** (update `backend/docs/ROLLOUT.md` Stage 5):
   - Before switching on, run `SELECT status, count(*) FROM company_subscription GROUP BY status`. Expect FREE_ACTIVE only and no TERMS_PENDING rows.
   - Set the Stripe keys and `STRIPE_WEBHOOK_SECRET`, register the webhook endpoint in Stripe, and set the Fakturownia keys.
   - Insert the pilot company IDs, then set `APP_PAYMENTS_PILOT_ENABLED=true` and `APP_PAYMENTS_ENABLED=true` together. Do it on a Stripe test key first: trial → upgrade → `invoice.paid` → invoice sent.
   - **Rollback:** cancel the pilot companies' Stripe subscriptions so the `customer.subscription.deleted` webhooks move them back to FREE, then set the flag to `false`. `PaymentsDisabledBootGuard` blocks startup until that has happened.
10. **Widen later.** Add companies to the table. General availability means setting `pilot.enabled=false`; the table can then be dropped.

## Risks and invariants
- **Companies outside the pilot are unchanged.**
  - No trial offer or paid actions in the UI.
  - Trial, upgrade and subscription consent are refused by the server.
  - FREE rollover and the campaign limit are unchanged.
  - The grace cron does not suspend them.

  Tests: extend `SubscriptionService_PaymentsToggleUnitTest` with enabled + pilot on + company not listed → `PaymentsDisabledException` and `trialEligible=false`. Add a cron unit test showing an unlisted `TERMS_PENDING` row is skipped. Update `plan-billing.component.spec.ts` and `layout.component.spec.ts` for `paymentsAvailable=false`. The `@payments-pilot` feature logs in as a company outside the pilot and checks that upgrade is refused.
- **Settling a payment is never blocked by the allow-list.** Webhooks must process events for any company with Stripe IDs, including one removed mid-subscription. If they didn't, a renewal or failure would go unrecorded and billing would drift from the company's actual plan. Tests: `SubscriptionService_WebhookUnitTest` / `StripeWebhookHandlerUnitTest` with a company not on the list.
- **Webhooks stay idempotent and use the existing locking.** Duplicate Stripe events are skipped by event ID and by a unique constraint, the subscription row has an optimistic-lock version, and lock conflicts return 503 so Stripe retries. None of this changes. Existing guards: `StripeWebhookHandlerUnitTest`, `StripeWebhookControllerUnitTest`, `SubscriptionService_Webhook_IntegrationTest`.
- **Consent before payment.** Subscription consent (`/subscription/consent`) must stay gated no more loosely than upgrade, so consent is never recorded for a plan the company cannot buy.
- **A mismatch between data and configuration stops startup.** Both guards stay: `PaymentsDisabledBootGuardUnitTest` plus a new `PaymentsPilotBootGuardUnitTest`. Also extend `PaymentsTogglePresence_IntegrationTest`: with enabled + pilot on, all the gated components exist and the disabled guard does not load.
- **Existing behaviour without the allow-list.** `pilot.enabled` defaults to `false`, so all current `enabled=true` test contexts and `payments-off.feature` / `RunPaymentsDisabledIT` must pass unchanged. Run them as the regression check.
- **Frontend fallback.** The public-config fetch still falls back to `true`. Because the server enforces eligibility, the worst case is a paid button that returns an error, never a charge.
- **Invoicing.** The Fakturownia retry cron and listener only act on invoice records created by `invoice.paid`, so companies outside the pilot produce none. Check the Fakturownia credentials before switching on; `FakturowniaAdapterUnitTest` and `InvoiceRetryServiceUnitTest` cover the adapter.
- **Stripe price IDs** in `application.yml:638-640` must match the live Stripe account. `resolvePlanNameFromPriceId` throws on unknown price IDs.

## Evidence
**Read in the code (FACT):**
- **FACT:** The flag defaults to `true` in Java but is `${APP_PAYMENTS_ENABLED:false}` in `application.yml`, and the Java comment says turning it back on must need no code changes (`AppPaymentsProperties.java:7-27`, `application.yml:410-411`).
- **FACT:** `SubscriptionPaidController`, `StripeWebhookController`, `StripeConfig`, `InvoiceRetryCronJob`, `TrialExpiryNotifierCronJob` and `InvoiceCreatedEventListener` carry `@ConditionalOnProperty(app.payments.enabled=true)` (read in each file, and confirmed by a Grep over the `subscription` package).
- **FACT:** `PaymentsDisabledBootGuard` loads only when the flag is `false` and throws if any trial, Business, Enterprise, downgrade-pending, payment-failed or terms-pending rows exist (`PaymentsDisabledBootGuard.java:24-57`).
- **FACT:** `requirePaymentsEnabled()` checks only the global flag and guards trial, upgrade, all webhook handlers, downgrade and cancel-downgrade (`SubscriptionService.java:45-49`, plus Grep for call sites at 61, 137, 208, 261, 317, 341, 381, 443, 513).
- **FACT:** `getStatus` builds `trialEligible` from `isTrialEligible`, which checks trial-used, FREE_ACTIVE and no Stripe customer ever, with no per-company payments check (`SubscriptionService.java:103-129, 750-758`).
- **FACT:** `processExpiredGracePeriods` suspends every expired `TERMS_PENDING` row, cancelling Stripe objects if present, and the query has no plan filter. `TermsGraceProcessorCronJob` skips this only when the flag is off (`SubscriptionService.java:855-887`, `CompanySubscriptionRepository.java:27-29`, `TermsGraceProcessorCronJob.java:37`).
- **FACT:** The trial-expiry and downgrade steps of the period cron run only when the flag is on; the FREE rollover always runs (`SubscriptionPeriodProcessorCronJob.java:36-41`).
- **FACT:** In main code, `enterTermsPending()` is called only from `TestSubscriptionController`, which is limited to the `(e2e | dev) & !prod & !test` profiles (Grep over `backend/src/main`; `TestSubscriptionController.java:38`).
- **FACT:** The webhook handler returns 503 on lock conflicts so Stripe retries, skips already-processed events, and routes five event types (`StripeWebhookController.java:49-64`, `StripeWebhookHandler.java:41-76`).
- **FACT:** Paid endpoints use the caller's own user ID (`SubscriptionPaidController.java:114-119`). The portal requires an existing Stripe customer (lines 98-101).
- **FACT:** `/public-config` is anonymous, not cached, and returns only the global flag (`PublicConfigController.java:12-35`). The frontend caches it and falls back to `true` (`public-config.service.ts:21-25`).
- **FACT:** The frontend hides paid actions with `paymentsEnabled()` at `plan-billing.component.html:215,264,275,302,355`. The layout calls `/subscription/status` for the trial offer only when payments are on and the user is a COMPANY (`layout.component.ts:254-268`).
- **FACT:** The rollout changeset reset every subscription to FREE and cleared the Stripe IDs (`09-04-2026-downgrade-all-to-free-for-rollout.sql:36-48`).
- **FACT:** The payments-off e2e feature checks public-config is false, paid endpoints return 401 to anonymous callers, and the webhook returns 404 (`payments-off.feature`). It runs in a separate Failsafe fork (`pom.xml:1913-1935`, from Grep).
- **FACT:** `StripeService` has no flag check, and `StripeDevEventPoller` is limited to the `dev-poller` profile (`StripeService.java`, `StripeDevEventPoller.java:33`).
- **FACT:** `FakturowniaConfig` has no startup condition (`FakturowniaConfig.java:11-22`). `fakturownia.enabled: true` is set in `application.yml:645-646`.

**Reasoned from the code (INFERENCE):**
- **INFERENCE:** Unless step 5 is done, switching the flag on is the one change that affects companies outside the pilot, because the grace cron starts running for all companies. The risk is small in production, since only a test-profile controller moves rows into `TERMS_PENDING`, but it isn't zero (SQL, or future code). Hence the pre-flight count and the scoped cron.
- **INFERENCE:** Checking eligibility only where payment starts is enough to keep companies outside the pilot out of paid states. Every paid or in-progress state is reached through trial activation, upgrade checkout, or webhooks that need a Stripe customer, and that customer is created only in `initiateUpgrade`.
- **INFERENCE:** The Liquibase downgrade changeset will not run again when the flag is turned on, because changesets run once per changeset ID.

**Not checked (HYPOTHESIS):**
- **HYPOTHESIS:** `PaymentsDisabledException` is turned into a translated 4xx by `common/exceptions/handlers/BusinessExceptionHandler.java`. To check, read its handler method before relying on the frontend's error text.
- **HYPOTHESIS:** `SubscriptionController` is also hidden from OpenAPI, so adding `paymentsAvailable` means a hand edit to `core/api-frozen/hidden-models.ts` rather than regenerating the client. To check, grep `@Hidden` in `SubscriptionController.java` and look for `SubscriptionStatusDtoOut` in `frontend/src/app/api`.
- **HYPOTHESIS:** Company team members, if they can open plan and billing, act under their own user ID and not the owning company's. Then the pilot table must hold the ID that `resolveUserId` returns. To check, read the team module and `PermissionUtils.getUserId`.
- **HYPOTHESIS:** The master Liquibase changelog includes the dated SQL files by path. To check, open `db/changelog/db.changelog-master.*` before adding the changeset.
