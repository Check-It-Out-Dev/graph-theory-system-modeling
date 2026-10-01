## Problem
Paid subscriptions are controlled by a single deployment-wide switch, `app.payments.enabled`. The goal is to turn on Stripe checkout, trials, downgrades, webhooks, invoicing and the paid-plan screens for a named set of pilot companies only. Every other company must behave exactly as it does now: only the free plan, no calls to Stripe or Fakturownia, no paid buttons, and no changes to its subscription rows. It's done when a pilot company can go through trial → upgrade → payment → invoice on a Stripe test key, a non-pilot company can't reach any paid path (in the API or the UI), and startup refuses to run if a non-pilot company has a paid subscription in progress.

## Where it lives today
**Backend, subsystem [11] (subscriptions and payments), package `com.sm.instagram.platform.subscription`, under `backend/src/main/java/.../platform/`:**

| Concern | File | How the switch acts |
|---|---|---|
| Flag | `resources/application.yml:410-411` (`APP_PAYMENTS_ENABLED:false`), `subscription/config/AppPaymentsProperties.java` (Java default `true`) | global boolean |
| Startup | `subscription/config/PaymentsDisabledBootGuard.java` | loads only when the flag is `false`. It refuses to start if any row is `TRIAL_ENTERPRISE`, `BUSINESS_ACTIVE`, `ENTERPRISE_ACTIVE`, `DOWNGRADE_PENDING`, `PAYMENT_FAILED` or `TERMS_PENDING` |
| Stripe SDK | `subscription/stripe/StripeConfig.java` (only loads when on), `StripePropertiesConfig.java` (always loads) | bean switch |
| Paid endpoints | `subscription/SubscriptionPaidController.java`: trial/activate, consent, upgrade, downgrade, downgrade/cancel, portal, config | the whole controller only loads when on (404 otherwise). `/portal` calls `StripeService` directly, without going through the service guard |
| Always-on reads | `subscription/SubscriptionController.java` (`/status`, `/invoices`) | no gate |
| Service guard | `SubscriptionService.requirePaymentsEnabled()`, called in `activateTrial`, `initiateUpgrade`, the `handleCheckoutCompleted`, `handleInvoicePaid`, `handlePaymentFailed`, `handleSubscriptionDeleted` and `handleSubscriptionUpdated` webhook handlers, `requestDowngrade` and `cancelDowngrade` | throws `PaymentsDisabledException` |
| Webhooks | `subscription/stripe/StripeWebhookController.java` | bean switch |
| Invoicing | `subscription/event/InvoiceCreatedEventListener.java`, `subscription/cron/InvoiceRetryCronJob.java` (every 15 min) | bean switch |
| Crons | `TrialExpiryNotifierCronJob` (bean switch); `SubscriptionPeriodProcessorCronJob` (the trial and downgrade branches check the flag inline, the free-period renewal always runs); `TermsGraceProcessorCronJob` (skips everything when off) | mixed |
| Public config | `common/publicconfig/PublicConfigController.java` and `PublicConfigDto(boolean paymentsEnabled)`: anonymous, `no-store` | global |
| Data | `CompanySubscription`: one row per company user (`user_id` is unique). Migration `09-04-2026-downgrade-all-to-free-for-rollout.sql` moved every row to `FREE_ACTIVE` | |

Relevant bulk jobs:
- `processExpiredTrials`, `processExpiredDowngrades`, `processExpiredGracePeriods` and `sendTrialEndingReminders` each run over all companies with no company filter.
- `enterTermsPending()` moves every active row, free ones included, to `TERMS_PENDING`. Its only caller is `TestSubscriptionController`, which runs only in the e2e and dev profiles.

**Frontend: [174] plan and billing, [177] shell and generated client**
- `core/config/public-config.service.ts` caches `paymentsEnabled` once per app and **shows** the paid actions if the fetch fails.
- `feature/plan-billing/plan-billing.component.ts:151` turns that into a signal. The template uses it at lines 215 (trial), 264 (free-only notice), 275 (cancel trial), 302 (Stripe portal) and 355 (paid actions).
- `layout/layout.component.ts:249-269` shows a trial-offer nudge to COMPANY users. It checks public config first, so a free-only deployment never calls `/subscription/status`.
- `core/subscription/subscription.service.ts` and `core/api-frozen/subscription.client.ts` are hand-maintained, because the paid controller is left out of the OpenAPI spec. `api/model/public-config-dto.ts` is generated.
- `demo-fixtures.ts:1619` and `sandbox/fixtures/plan-billing.fixture.ts` hold fake values for the flag.

**Tests and docs that pin today's behaviour**
- Backend: `SubscriptionService_PaymentsToggleUnitTest`, `PaymentsTogglePresence_IntegrationTest`, `SubscriptionService_PaymentsDisabled_IntegrationTest`, `PublicConfigController{Unit,_Integration}Test`, `RunPaymentsDisabledIT` with `features/payments_off/payments-off.feature`, and the flag values in the test `application*.properties` files.
- Frontend: `plan-billing.component.spec.ts`, `layout.component.spec.ts`, e2e `payments-off.spec.ts`, `subscription-lifecycle.spec.ts`, `bdd/payments-off.feature`.
- Docs: `backend/docs/ROLLOUT.md` stage 5.

## Proposed change
Split the switch in two:
1. **The infrastructure switch stays global.** `app.payments.enabled` still decides whether the Stripe SDK, the webhook, the invoicing listener and the paid crons exist. Webhooks have to arrive at one endpoint per Stripe account, so this can't be per company.
2. **Paid access is decided per company.** Add a new setting, `app.payments.pilot-only` (defaults to `true` in `application.yml`), and a boolean column `company_subscription.payments_pilot` (default `false`). A new `PaymentsEligibility` service answers: `isEnabledFor(sub) = enabled && (!pilotOnly || sub.paymentsPilot)`.

**Where to store the pilot list.** I considered two options:
- **An environment list of user IDs:** simpler, but every change needs a restart, and the boot guard couldn't check it against the data.
- **A column on the one-row-per-company subscription table (chosen):** it can be locked together with the status, checked by the boot guard in one query, and changed without a redeploy. Each change is written to `SubscriptionEvent`.

**How each part changes:**
- **Actions a company starts** (trial, upgrade, downgrade, cancel downgrade, portal, consent) check the company flag. Otherwise they return 403 or `PaymentsDisabledException`, never 404.
- **Webhook handlers keep the global check only.** Only pilot companies can get a Stripe customer or subscription ID, and if a company's pilot flag is later removed, its money events must still be recorded.
- **Crons** in pilot-only mode handle only pilot rows. The free-period renewal still runs for everyone.
- **Boot guard:** replace it with `PaymentsBootGuard`, which always loads. With payments off it keeps today's check. In pilot-only mode it refuses to start if a **non-pilot** row is in a paid status or has a Stripe ID.
- **Frontend:** `PublicConfigDto` gets `paymentsMode: OFF | PILOT | ON`. `paymentsEnabled` is kept as `mode == ON`, so anonymous pricing stays hidden during the pilot. `SubscriptionStatusDtoOut` gets `paymentsEnabled` for the logged-in company. The billing screens use the per-company value, and only skip the status call when the mode is `OFF`.

## Plan
Every step is safe to ship on its own while production still has `APP_PAYMENTS_ENABLED=false`, because the global check still runs first.

1. **Schema.** Add a Liquibase changeset for `company_subscription.payments_pilot BOOLEAN NOT NULL DEFAULT false`, plus the field on `CompanySubscription.java`. Add `SubscriptionEventType.PAYMENTS_PILOT_GRANTED/REVOKED` (check whether the column has a DB constraint).
2. **Setting and eligibility service.** Add `pilotOnly` to `AppPaymentsProperties.java` and `pilot-only: ${APP_PAYMENTS_PILOT_ONLY:true}` to `application.yml`. Create `subscription/config/PaymentsEligibility.java`. Set `pilot-only=false` explicitly in the test `application*.properties` files where the full-on suites expect it.
3. **Service guards.** In `SubscriptionService.java`, replace `requirePaymentsEnabled()` with `requirePaymentsEnabledFor(subscription)` in `activateTrial`, `initiateUpgrade`, `requestDowngrade` and `cancelDowngrade`. Leave the global check in the five webhook handlers. In `SubscriptionPaidController.java`, add the per-company check to `/portal` and `/consent`.
4. **Crons.** In pilot-only mode, filter the rows in `processExpiredTrials`, `processExpiredDowngrades`, `processExpiredGracePeriods`, `processTrialExpiryInTermsPending` and `sendTrialEndingReminders` to `paymentsPilot = true`. The simplest way is `...AndPaymentsPilotTrue` query variants in `CompanySubscriptionRepository.java`. The crons themselves don't change, and `renewExpiredFreeBillingPeriods` keeps running for everyone.
5. **Boot guard.** Rename `PaymentsDisabledBootGuard.java` to `PaymentsBootGuard.java` and remove its bean condition. Keep the existing branch for payments off. In pilot-only mode, count non-pilot rows in paid statuses, or with `stripeCustomerId` or `stripeSubscriptionId` set, and refuse to start if there are any. Add the repository query.
6. **Managing the pilot list.** Add an admin-only endpoint (next to the other admin controllers) or a documented SQL runbook to grant or revoke the flag. Each change is written to `SubscriptionEvent`. **Revoking is refused** unless the row is `FREE_ACTIVE` with no Stripe subscription or schedule.
7. **Public contract.** Add `paymentsMode` to `PublicConfigDto.java` and set it in `PublicConfigController.java`. Add `paymentsEnabled` to `SubscriptionStatusDtoOut`, set in `getStatus`. Regenerate `backend/docs/openapi/openapi.json`, then run the frontend `openapi:gen` (updates `api/model/public-config-dto.ts` and `frontend/docs/openapi/openapi.json`) and `tools/check-contract-coverage.mjs`.
8. **Frontend.**
   - `core/api-frozen/subscription.client.ts` / `hidden-models.ts`: add the status field.
   - `core/config/public-config.service.ts`: expose `paymentsMode()`.
   - `plan-billing.component.ts` and `.html`: use `status.paymentsEnabled`.
   - `layout.component.ts:254-268`: skip only when the mode is `OFF`, then check `sub.paymentsEnabled`.
   - Update `demo-fixtures.ts` and `sandbox/fixtures/plan-billing.fixture.ts`.
9. **Tests** (see below). Update `ROLLOUT.md` stage 5 and `DEV-LITE.md`.
10. **Rollout.** On staging with a Stripe test key: set `ENABLED=true` and `PILOT_ONLY=true`, register the webhook and secret, set `fakturownia.enabled` and its keys, then grant one company. Run trial → upgrade → payment → invoice for the pilot company and the payments-off checks for a non-pilot one. Then do the same in production with live keys.

## Risks and invariants
| Invariant | Guarding test |
|---|---|
| **I1** A non-pilot company can't create a Stripe customer or session, or enter a paid status, through any endpoint | new `SubscriptionService_PaymentsPilotUnitTest` (every user-started method, both pilot and non-pilot) and a `payments-pilot.feature` on the backend Cucumber runner (like `RunPaymentsDisabledIT`) |
| **I2** With `enabled=false`, behaviour is byte-for-byte today's | existing `PaymentsTogglePresence_IntegrationTest`, `SubscriptionService_PaymentsDisabled_IntegrationTest`, `payments-off.feature`, frontend `payments-off.spec.ts`, all unchanged and green |
| **I3** Money events are never dropped: webhooks for a company with a Stripe link are processed even if its flag changed; revoking needs `FREE_ACTIVE` and no Stripe IDs | unit test on the revoke rule; webhook test with a revoked-but-linked row |
| **I4** Startup check: in pilot-only mode, no non-pilot row is in a paid status or has Stripe IDs | new cases in the boot-guard integration test (a non-pilot `BUSINESS_ACTIVE` row fails startup; a pilot one passes) |
| **I5** Crons never change non-pilot rows (e.g. `processExpiredGracePeriods` → `SUSPENDED_LEGAL`, campaign limit 0); free-period renewal still runs for all | extend `SubscriptionService_CronUnitTest` |
| **I6** Webhook idempotency and optimistic-lock handling (503 on a lock conflict, so Stripe retries) unchanged | existing `StripeWebhookControllerUnitTest` and `SubscriptionService_WebhookUnitTest` |
| **I7** Consent: paid actions still need subscription consent (`LegalConsentService`); terms-version handling unchanged | existing consent and terms tests |
| **I8** Anonymous pricing stays hidden during the pilot; the frontend never shows a paid button to a non-pilot company | `PublicConfigController` tests for all three modes; `plan-billing.component.spec.ts` and `layout.component.spec.ts` with pilot and non-pilot status; e2e `subscription-lifecycle.spec.ts` run as a pilot company |

Failure modes:
- **Misconfiguration.** Setting `ENABLED=true` without `PILOT_ONLY` opens paid plans to everyone if the Java default is used. That's why the yml default is `true`, and startup should log the mode at INFO.
- **Frontend safe default.** Today a failed public-config fetch falls back to `true`. In `PILOT` mode the per-company status decides instead, so a failed status call must hide the paid actions.
- **Invoicing (Fakturownia).** The new per-company gate doesn't cover invoicing. Its own `fakturownia.enabled` setting (in addition to the global switch) controls it.

## Evidence
- FACT: the flag is defined in `application.yml:411` (default false); `AppPaymentsProperties.enabled` defaults to `true` in Java (both files read).
- FACT: `PaymentsDisabledBootGuard` loads only when the flag is `false`, and refuses to start on the six statuses listed (read).
- FACT: `SubscriptionPaidController`, `StripeWebhookController`, `StripeConfig`, `InvoiceRetryCronJob`, `InvoiceCreatedEventListener` and `TrialExpiryNotifierCronJob` only load when the flag is on (grep of `@ConditionalOnProperty`).
- FACT: `/portal` calls `stripeService.createPortalSession` without the service guard (`SubscriptionPaidController.java:95-107` read).
- FACT: the period cron gates its trial and downgrade branches inline and always renews free periods; the grace cron skips everything when off (both read).
- FACT: `processExpiredTrials`, `processExpiredDowngrades` and `processExpiredGracePeriods` loop over all matching rows with no company filter; grace expiry sets `SUSPENDED_LEGAL` and cancels Stripe (`SubscriptionService.java:557-622, 856-887` read).
- FACT: `CompanySubscription` has a unique `user_id`, so there is one row per company user (grep of the entity).
- FACT: `enterTermsPending()` has only one caller, `TestSubscriptionController`, which is `@Profile("(e2e | dev) & !prod & !test")` (grep).
- FACT: `PublicConfigController` is anonymous and `no-store`; the frontend service falls back to `true` on error and caches with `shareReplay` (both read).
- FACT: the layout nudge short-circuits on public config before calling `getStatus` (`layout.component.ts:249-269` read); the billing template uses `paymentsEnabled()` at lines 215, 264, 275, 302 and 355 (grep).
- FACT: `SubscriptionPaidController` is `@Hidden`, so it isn't in the OpenAPI spec (read).
- INFERENCE: webhooks can only concern pilot rows, because Stripe customer IDs are created only in `initiateUpgrade`, which will be gated per company. Lookups use `findByStripeCustomerIdForUpdate` / `findByStripeSubscriptionIdForUpdate` (read). *Not verified: that no other code path sets a Stripe customer ID.*
- HYPOTHESIS: `InvoiceRetryService` works only on `InvoiceRecord` rows created by paid webhooks, and Fakturownia calls are separately gated by `fakturownia.enabled`.
- HYPOTHESIS: the downgrade-all-to-free migration is a one-time changeset (not `runAlways` or `runOnChange`), so it won't reset pilot companies.
- HYPOTHESIS: the frontend status model lives in the hand-maintained frozen client, not the generated one.

## Corrections after verification

## What the checks found
- **One-time migration:** `09-04-2026-downgrade-all-to-free-for-rollout.sql` is a plain changeset (`system:downgrade-all-to-free-for-rollout`) with no `runAlways` or `runOnChange`. It won't run again and reset pilot companies. The HYPOTHESIS becomes a FACT.
- **Stripe customer IDs:** in production code, `setStripeCustomerId` is called only in `SubscriptionService.java:158`, inside `initiateUpgrade`. The only other callers are in `TestSubscriptionController`, which runs only in the e2e and dev profiles. So the INFERENCE holds: once upgrade is gated per company, webhooks can only concern pilot rows.
- **Invoicing:** `FakturowniaAdapter.java:45` refuses to call Fakturownia unless `properties.isConfigured()` is true, meaning `enabled` (Java default `true`) and a non-blank API key. `InvoiceRetryService` only processes `InvoiceRecord` rows returned by `findRetryable`. FACT, with one correction below.
- **Frontend status model:** `trialEligible` and `hasStripeSubscription` are defined in `core/api-frozen/hidden-models.ts`, besides the demo fixtures. Step 8 should edit `hidden-models.ts` (and `subscription.client.ts` only if it re-declares the shape), not the generated client. FACT.

## Corrections
1. **Invoicing gate.** In "Risks and invariants", Fakturownia's own gate is not just `fakturownia.enabled`: it's `fakturownia.enabled` **and** a non-blank `FAKTUROWNIA_API_KEY`. `enabled` defaults to `true` in Java, so in practice the API key decides whether invoices are sent. Invoicing is not gated per company. It stays pilot-only because invoice records only come from paid webhooks.
2. **Frontend file.** In Plan step 8, the file to extend with `paymentsEnabled` is `frontend/src/app/core/api-frozen/hidden-models.ts`.
