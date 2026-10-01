# Paid subscriptions for a pilot set of companies

## Problem

`app.payments.enabled` is one switch for the whole platform, and it's off by default (`${APP_PAYMENTS_ENABLED:false}`). When it's off, all paid infrastructure is missing from the running app: the paid controllers, the Stripe webhook, the Stripe SDK setup, trial reminders, the invoice listener and the invoice retry job. A boot guard also refuses to start the app while any company is in a paid state. The switch can't express "on for these companies only". Turning it on for the pilot would turn it on for every company, and it would also restart invoicing jobs across the whole database.

We need paid subscriptions (trial, upgrade, downgrade, portal, webhooks, invoicing) for a named set of companies. Every other company should see and do exactly what it does today. (A "company" here is a `User` with the `COMPANY` authority; a subscription is one `company_subscription` row per `user_id`.)

## Where it lives today

**Backend: `backend/src/main/java/com/sm/instagram/platform/subscription/`**

| Concern | File | How the switch applies |
|---|---|---|
| The flag | `config/AppPaymentsProperties.java`, `config/AppPaymentsConfiguration.java`, `src/main/resources/application.yml:410-411` | `app.payments.enabled`. The Java field defaults to `true`, but yml defaults to `false`. |
| Startup | `config/PaymentsDisabledBootGuard.java` | Loaded only when the flag is `false`. Refuses to boot if any row is `TRIAL_ENTERPRISE`, `BUSINESS_ACTIVE`, `ENTERPRISE_ACTIVE`, `DOWNGRADE_PENDING`, `PAYMENT_FAILED` or `TERMS_PENDING`. |
| Stripe SDK | `stripe/StripeConfig.java` (loaded only when on), `stripe/StripePropertiesConfig.java` + `stripe/StripeService.java` (always loaded, inactive when off) | `Stripe.apiKey` is set only when the flag is on. |
| Paid endpoints | `SubscriptionPaidController.java` (loaded only when on): `/trial/activate`, `/consent`, `/upgrade`, `/downgrade`, `/downgrade/cancel`, `/portal`, `/config` | Missing from the app when off, so they return 404. |
| Always-on endpoints | `SubscriptionController.java` (`/status`, `/invoices`), `common/publicconfig/PublicConfigController.java` + `PublicConfigDto(boolean paymentsEnabled)` | The anonymous `/public-config` endpoint returns the global flag with `no-store`. |
| Webhooks | `stripe/StripeWebhookController.java` (loaded only when on) → `stripe/StripeWebhookHandler.java` (always loaded) → `SubscriptionService.handle*` | Security allows the webhook path without login (`WebSecurityConfiguration.java:154`, `JwtAuthenticationFilter.java:552`). |
| Service guard | `SubscriptionService.requirePaymentsEnabled()`, called by `activateTrial`, `initiateUpgrade`, the 5 `handle*` webhook methods, `requestDowngrade` and `cancelDowngrade` | Throws `PaymentsDisabledException`, which becomes **HTTP 503** (`common/exceptions/handlers/BusinessExceptionHandler.java:121-144`). |
| Scheduled jobs | `cron/SubscriptionPeriodProcessorCronJob.java` (always loaded; trial/downgrade branches only when on; FREE renewal always runs) | |
| | `cron/TermsGraceProcessorCronJob.java` (always loaded; does nothing when off) | |
| | `cron/TrialExpiryNotifierCronJob.java`, `cron/InvoiceRetryCronJob.java` (loaded only when on) | |
| Invoicing | `event/InvoiceCreatedEventListener.java` (loaded only when on), `invoicing/InvoiceRetryService.java`, `invoicing/FakturowniaAdapter.java`, `yml fakturownia.enabled: true` | |
| Data | `db/changelog/2026/04/09-04-2026-downgrade-all-to-free-for-rollout.sql` (moved every row to FREE and cleared Stripe IDs; invoice rows were left alone) | |
| | `09-04-2026-update-free-plan-campaign-limit.sql` (FREE limit 2 → 5); `2026/03/22-03-2026-subscription-tables.sql:192-195` (BUSINESS 29 PLN / 5 campaigns, ENTERPRISE 99 PLN / 10) | |
| Test/dev only | `TestSubscriptionController.java` (profiles `e2e \| dev`), `stripe/StripeDevEventPoller.java` (profile `dev-poller`, not tied to the flag) | |
| Runbook | `backend/docs/ROLLOUT.md` Stage 5 | |

**Frontend: `frontend/src/app/`**
- `core/config/public-config.service.ts`: fetches `paymentsEnabled` once and caches it. If the fetch fails it assumes **true**.
- `feature/plan-billing/plan-billing.component.ts` and `.html`: `paymentsEnabled` starts as `true`. It controls the trial offer (html:215), the "payments disabled" notice (264), cancel-trial (275), the portal row (302) and the plan cards (355). The PAYMENT_FAILED banner (143) and the pending-downgrade banner (172) don't check the flag; they depend only on the subscription status.
- `layout/layout.component.ts:254-268`: the trial nudge for companies checks the public config first. When payments are off, it never calls `/subscription/status`.
- `core/subscription/subscription.service.ts`: `SubscriptionApiService` (read) and `SubscriptionWriteApi` (paid calls). The DTOs are hand-kept in `core/api-frozen/hidden-models`, because the backend controllers are `@Hidden` from OpenAPI.
- `app.routes.ts:596-606`: `/subscription/success` and `/subscription/cancel` (Stripe return URLs) redirect to `/user/settings/plan-billing`.
- `plan-billing.component.ts:72-106`: the plan list shows **FREE = 2, BUSINESS = 5, ENTERPRISE = 10** campaigns.
- Demo and sandbox data: `core/demo/demo-fixtures.ts:1619`, `sandbox/fixtures/plan-billing.fixture.ts`.

**Tests that pin today's behaviour**
- Backend: `integration/service/subscription/PaymentsTogglePresence_IntegrationTest.java` (which beans exist when on vs off), `SubscriptionService_PaymentsDisabled_IntegrationTest.java`, `unit/service/subscription/SubscriptionService_PaymentsToggleUnitTest.java`, `integration/controller/PublicConfigController_IntegrationTest.java`, `unit/controller/publicconfig/PublicConfigControllerUnitTest.java`, `e2e/RunPaymentsDisabledIT.java` + `features/payments_off/payments-off.feature` (a separate Failsafe run in `pom.xml:1912-1939` with `-Dapp.payments.enabled=false`). The test property files set the flag to `true`.
- Frontend: `plan-billing.component.spec.ts`, `layout.component.spec.ts`, and e2e `e2e-tests/_framework/api/{payments-config,subscription}.api.ts`, `bdd/steps/subscription.steps.ts`.

**Flows that matter**
1. **Upgrade:** `/upgrade` → `initiateUpgrade` creates a Stripe customer and a Checkout Session → Stripe sends `checkout.session.completed` → `handleCheckoutCompleted` looks up the row by `stripeCustomerId` → `invoice.paid` → `handleInvoicePaid` creates the `InvoiceRecord` and publishes `InvoiceCreatedEvent` → the listener sends it to Fakturownia → `InvoiceRetryCronJob` retries anything still `PENDING`/`FAILED`.
2. **Webhooks match on Stripe IDs** (`stripeCustomerId`/`stripeSubscriptionId`), never on the logged-in user.
3. **Jobs work in bulk across all companies**, not per company.

## Proposed change

**The design separates "is the payment machinery running" from "is this company allowed to pay".**

1. **Keep `app.payments.enabled` as the machinery switch** and add `app.payments.mode: ALL | PILOT`, defaulting to `ALL`.
   - `enabled=false`: exactly today's behaviour.
   - `enabled=true, mode=ALL`: exactly today's "on" behaviour. This keeps the documented promise that flipping the flag restores everything with no code changes, and every existing "on" test context keeps working.
   - `enabled=true, mode=PILOT`: all the payment components load (Stripe, webhook, jobs, invoicing), but only pilot companies can *start* anything paid.

2. **Store pilot membership in the database**, in a new table `payments_pilot_company(user_id PK/FK, enabled_at, enabled_by, note)` created by a Liquibase changeset.
   - Not a config list of IDs: a DB table can be changed without a redeploy and keeps an audit trail.
   - Not a column on `company_subscription`: that row has `@Version` optimistic locking and gets created lazily, and we don't want to write to non-pilot rows.

3. **Add one decision point, `PaymentsAccessPolicy`** (in `subscription/config/`):
   - `isInfrastructureOn()` = the flag.
   - `isAvailableFor(userId)` = `enabled && (mode == ALL || pilotRepo.existsById(userId))`.
   - Everything else asks this class. Nobody reads `AppPaymentsProperties` directly any more.

4. **Company-initiated actions check the company; Stripe-initiated actions don't.**
   - **User actions** check `isAvailableFor(userId)`: `activateTrial`, `initiateUpgrade`, `requestDowngrade`, `cancelDowngrade`, and in the controller `/consent`, `/portal` and `/config`. Today `/consent`, `/portal` and `/config` have no service check at all.
   - A non-pilot company gets a new `PaymentsNotAvailableException` → **404**. That matches what these companies see today (the endpoint doesn't exist). The existing 503 would look like an outage.
   - **Webhook handlers** keep only the machinery check. A company that was removed from the pilot, or whose Stripe state changes on Stripe's side, **must** still have `invoice.payment_failed` and `customer.subscription.deleted` processed. Otherwise its row is stranded, which is exactly what the existing boot guard exists to prevent. This is safe because a Stripe customer can only be created by the pilot-gated `initiateUpgrade`, so non-pilot rows have no Stripe IDs for a webhook to match.

5. **Scheduled jobs:** restrict the bulk queries that could reach non-pilot companies in PILOT mode.
   - `processExpiredGracePeriods` and `processTrialExpiryInTermsPending` get a pilot filter. When the flag is on, `TermsGraceProcessorCronJob` stops doing nothing and would otherwise suspend *any* `TERMS_PENDING` company to `SUSPENDED_LEGAL` (campaign limit 0).
   - `processExpiredTrials`, `processExpiredDowngrades` and trial reminders only touch statuses that non-pilot companies can't reach, so they stay as they are, protected by the new boot guard.
   - FREE period renewal stays global, as today.

6. **Invoicing:** add a mandatory pre-flight step. The April migration reset subscriptions but did **not** touch `invoice_record`. `findRetryable` picks up *all* `PENDING`/`FAILED` rows under the retry limit. Turning the machinery on would re-send any old sandbox-era invoices to live Fakturownia within 15 minutes. So: count them, and move them to `DEAD_LETTER` with a note (by migration or runbook) before the flip. The retry service also gets a pilot filter in PILOT mode.

7. **Startup:** add `PaymentsPilotBootGuard`, loaded only when `enabled=true, mode=PILOT`. It refuses to boot if any row that is in a paid status, or that has a Stripe customer/subscription/schedule ID, belongs to a company outside the pilot.
   - Move `IN_FLIGHT_PAID_STATUSES` into a shared constant so both guards use the same set.
   - Removing a company from the pilot is refused while that company is in such a state. Removal happens only after its subscription has ended (it's back to `FREE_ACTIVE` with no Stripe IDs).

8. **What the frontend is told:**
   - `/public-config` stays anonymous and global. `paymentsEnabled` becomes `enabled && mode == ALL`, so it stays **`false` in PILOT mode**. The landing page, anonymous visitors and non-pilot companies see exactly what they see today, and the layout nudge still skips `/status` for everyone.
   - Add `paymentsAvailable` (per company) to `SubscriptionStatusDtoOut`. The plan-billing page uses `status.paymentsAvailable ?? publicConfig.paymentsEnabled`. An older backend without the field falls back to today's behaviour.
   - Pilot companies won't get the shell trial nudge. That's acceptable for a hand-picked pilot, and it avoids a `/status` call for every company (`getStatus` → `getOrCreateSubscription` creates a row as a side effect).

**Why this design:** it reuses the existing split between user actions and webhooks, never loads a component per company (Spring can't do that), and keeps all three existing modes exactly as they are. Each new restriction is a filter, so a mistake there shows up as a pilot company being refused, not as a stranger being charged.

## Plan

Each step can be shipped on its own. Production stays at `enabled=false` until step 9.

1. **Clarify the configuration.** Add `mode` (enum, default `ALL`) to `AppPaymentsProperties`. Make the Java default for `enabled` match yml (`false`), or at least document the mismatch with `matchIfMissing=false`. Add `mode: ${APP_PAYMENTS_MODE:ALL}` to yml.
   *Files:* `subscription/config/AppPaymentsProperties.java`, `src/main/resources/application.yml`, `src/test/resources/application*.properties`.
2. **Schema and repository.** Add a Liquibase changeset for `payments_pilot_company` (with rollback), a `PaymentsPilotCompany` entity and a repository.
   *Files:* `db/changelog/2026/09/…-payments-pilot-company.sql`, `db/changelog/changelog.xml`, `subscription/entity/PaymentsPilotCompany.java`, `subscription/repository/PaymentsPilotCompanyRepository.java`.
3. **Policy and exception.** Add `PaymentsAccessPolicy` and `PaymentsNotAvailableException` with a 404 handler. Move the in-flight status set into a shared constant.
   *Files:* `subscription/config/PaymentsAccessPolicy.java`, `subscription/exception/PaymentsNotAvailableException.java`, `common/exceptions/handlers/BusinessExceptionHandler.java`, `subscription/config/PaymentsDisabledBootGuard.java`, `subscription/entity/SubscriptionStatus.java` (or a new `PaidStatuses` holder).
4. **Service checks.** Replace `requirePaymentsEnabled()` with `requireAvailableFor(userId)` in the four user-initiated methods. Keep the machinery-only check in the five `handle*` methods. Add a pilot filter to `processExpiredGracePeriods`, `processTrialExpiryInTermsPending` and `InvoiceRetryService.retryFailedInvoices` in PILOT mode. Add `paymentsAvailable` to the status DTO.
   *Files:* `subscription/SubscriptionService.java`, `subscription/dto/SubscriptionStatusDtoOut.java`, `subscription/repository/CompanySubscriptionRepository.java`, `subscription/repository/InvoiceRecordRepository.java`, `subscription/invoicing/InvoiceRetryService.java`.
5. **Controller checks.** In `/consent`, `/portal` and `/config`, call the policy before doing anything. Make `/public-config` return `enabled && mode==ALL`.
   *Files:* `subscription/SubscriptionPaidController.java`, `common/publicconfig/PublicConfigController.java`.
6. **Pilot boot guard and job wiring.** Add `PaymentsPilotBootGuard` (loaded only for `enabled=true` + `mode=PILOT`). Have the job classes ask the policy instead of the properties.
   *Files:* `subscription/config/PaymentsPilotBootGuard.java`, `cron/TermsGraceProcessorCronJob.java`, `cron/SubscriptionPeriodProcessorCronJob.java`.
7. **Admin/runbook.** Minimum: SQL steps in the runbook to add or remove a company (removal refused while the company is in a paid state or has Stripe IDs). Optional: admin endpoints for the same with a permission check.
   *Files:* `backend/docs/ROLLOUT.md` (a new "Stage 5a — pilot" section), optionally `admin/AdminController.java`.
8. **Frontend.** Add `paymentsAvailable?: boolean` to the hidden status model. In plan-billing, compute the flag as `status()?.paymentsAvailable ?? publicConfigFlag()`. Leave the layout nudge unchanged, but add a comment about PILOT mode. Update the demo and sandbox data.
   *Files:* `core/api-frozen/hidden-models*`, `feature/plan-billing/plan-billing.component.ts` and `.html`, `core/config/public-config.service.ts` (doc comment), `core/demo/demo-fixtures.ts`, `sandbox/fixtures/plan-billing.fixture.ts`.
9. **Pre-flight, then the flip.** In this order:
   1. Confirm with product that the plan limits in the database match what's being sold (see risk 7).
   2. Check that the Stripe price IDs in yml and the seed are live-mode prices, and that `STRIPE_WEBHOOK_SECRET` has been rotated.
   3. Count `invoice_record` rows in `PENDING`/`FAILED`, and dead-letter them.
   4. Insert the pilot companies.
   5. Deploy with `APP_PAYMENTS_ENABLED=true` and `APP_PAYMENTS_MODE=PILOT`.
   6. Walk one pilot company through trial → upgrade → `invoice.paid` → Fakturownia on the Stripe test key first, as `ROLLOUT.md` Stage 5 says.

   Rollback is safe only once no pilot company is in a paid state; the existing boot guard enforces that.
10. **Tests** (see the next section).
    *Files:* the backend `test/java/...` classes listed there, `pom.xml` (new Failsafe run), a new `features/payments_pilot/payments-pilot.feature`, and frontend `plan-billing.component.spec.ts`, `layout.component.spec.ts`, `public-config.service.spec.ts`.

## Risks and invariants

| # | Invariant / risk | Guarding test (existing ✔ / new ✚) |
|---|---|---|
| I1 | With `enabled=false`, nothing changes: the paid components aren't loaded, the "off" boot guard is loaded, `/public-config` returns `false`, the webhook returns 404. | ✔ `PaymentsTogglePresence_IntegrationTest.WhenPaymentsDisabled`, ✔ `RunPaymentsDisabledIT` / `payments-off.feature`, ✔ `PublicConfigController_IntegrationTest` |
| I2 | With `enabled=true, mode=ALL`, today's "on" behaviour is unchanged, so every existing "on" suite passes without edits. | ✔ `PaymentsTogglePresence_IntegrationTest.WhenPaymentsEnabled`, ✔ `SubscriptionService*UnitTest` suites (stub `mode=ALL`) |
| I3 | A non-pilot company can't create a Stripe customer, a checkout session, a trial, a consent record or a portal session: 404, and no calls to `StripeService`. | ✚ `SubscriptionService_PaymentsPilotUnitTest` (same shape as `SubscriptionService_PaymentsToggleUnitTest`, including `verifyNoInteractions(stripeService)`), ✚ `SubscriptionPaidController_Pilot_IntegrationTest` |
| I4 | Webhooks are **never** filtered by pilot membership. Removing a company from the pilot can't leave its subscription stuck. | ✚ unit test: `handlePaymentFailed`/`handleSubscriptionDeleted` still process a linked row whose owner isn't in the pilot |
| I5 | Every row that is in a paid status or has Stripe IDs belongs to a pilot company (checked at boot in PILOT mode; removal refused otherwise). | ✚ `PaymentsPilotBootGuard_IntegrationTest` (passes with pilot rows, fails with a paid row outside the pilot), ✚ removal-refusal test |
| I6 | The grace-period and terms-pending jobs never move a non-pilot company to `SUSPENDED_LEGAL` in PILOT mode. FREE period renewal still runs for everyone. | ✚ `SubscriptionService_CronUnitTest` pilot cases, ✔ `SubscriptionService_PaymentsDisabled_IntegrationTest.FreeIdempotence` |
| I7 | Turning the machinery on doesn't send old invoices to Fakturownia; the retry job in PILOT mode only touches pilot companies' invoices. | ✚ `InvoiceRetryService` pilot-filter test, ✚ pre-flight query in the runbook |
| I8 | The anonymous `/public-config` still returns `false` in PILOT mode, stays `no-store`, and is reachable without login. | ✚ third nested class in `PublicConfigController_IntegrationTest` / `PublicConfigControllerUnitTest` |
| I9 | Frontend: a non-pilot company sees today's free-only page (no trial offer, plan cards or portal; the "payments disabled" notice is shown). A pilot company sees the paid actions even though `/public-config` is `false`. Missing `paymentsAvailable` falls back to the public flag. | ✚ `plan-billing.component.spec.ts` (3 cases), ✔ `layout.component.spec.ts` (no `/status` call when public config is `false`) |
| I10 | Existing webhook safety (duplicate-event checks via `existsByStripeEventId` + unique DB constraint, and 503 on optimistic-lock conflicts so Stripe retries) is unchanged. | ✔ `SubscriptionService_WebhookUnitTest` |

**Other risks**
- **R1 (Failsafe contexts):** PILOT mode needs its own Spring context. Add a `payments-pilot` Failsafe run (a copy of `pom.xml:1917-1939` with `app.payments.mode=PILOT`), or cover it only with `@TestPropertySource` integration tests. Don't mix it into the "on" run.
- **R2 (frontend error mapping):** `classifyTrialError` treats 409 as "trial already used". The new 404 falls into the generic error, which is acceptable, but it must not be 409.
- **R3 (dev poller):** `StripeDevEventPoller` runs under the `dev-poller` profile regardless of the flag. Leave it, but note it in the dev docs.
- **R4 (mode read without `enabled`):** reading `mode` without checking `enabled` would reopen the off path. The policy must always check `enabled` first (I1 covers this).
- **R5 (price and limit mismatch):** the database has FREE = 5 and BUSINESS = 5 campaigns, while the frontend plan list says FREE 2 / BUSINESS 5. A pilot company upgrading from FREE to BUSINESS would pay 29 PLN for **no extra campaigns**. Product must fix this (migration and/or frontend plan list) **before** anyone is charged.
- **R6 (migration rollback):** the April migration's rollback is a no-op (the paid states can't be reconstructed). Any pilot rollback relies on the boot guard and cancelling subscriptions on the Stripe side, not on Liquibase.

## Evidence

**Configuration and startup**
1. The global flag is `app.payments.enabled`, which defaults to `false` in yml, while the Java field defaults to `true`. **FACT**: `application.yml:410-411`; `AppPaymentsProperties.java:23-27`.
2. `SubscriptionPaidController`, `StripeWebhookController`, `StripeConfig`, `TrialExpiryNotifierCronJob`, `InvoiceRetryCronJob` and `InvoiceCreatedEventListener` load only when `havingValue="true", matchIfMissing=false`. **FACT**: the annotations in each file (e.g. `SubscriptionPaidController.java:47`, `StripeWebhookController.java:21`, `InvoiceRetryCronJob.java:22`).
3. The boot guard loads only when the flag is off and counts six in-flight paid statuses. **FACT**: `PaymentsDisabledBootGuard.java:24,35-41`.

**Service checks and webhooks**

4. `requirePaymentsEnabled()` is called in exactly `activateTrial`, `initiateUpgrade`, the 5 `handle*` methods, `requestDowngrade` and `cancelDowngrade`. `/consent`, `/portal` and `/config` have no service check. **FACT**: `SubscriptionService.java:61,137,208,261,317,341,381,443,513`; `SubscriptionPaidController.java:67-112`.
5. `PaymentsDisabledException` maps to HTTP 503. **FACT**: `BusinessExceptionHandler.java:121-144`.
6. Webhook handlers find rows by Stripe customer/subscription ID, and a Stripe customer is created only in `initiateUpgrade`. **FACT**: `SubscriptionService.java:147-160,209,265,318,342,382`. It follows that non-pilot companies can't be matched by webhooks. **INFERENCE** (it assumes no other code writes `stripeCustomerId`; only the e2e/dev `TestSubscriptionController.java:85,215` does).
7. The webhook handler checks for duplicate events and re-throws optimistic-lock conflicts; the controller returns 503 so Stripe retries. **FACT**: `StripeWebhookHandler.java:41-60`; `StripeWebhookController.java:51-58`.

**Scheduled jobs**

8. The period-processor job always renews FREE periods and runs trial/downgrade expiry only when the flag is on. The terms-grace job does nothing when the flag is off. **FACT**: `SubscriptionPeriodProcessorCronJob.java:36-41`; `TermsGraceProcessorCronJob.java:37`.
9. `processExpiredGracePeriods` moves any expired `TERMS_PENDING` row to `SUSPENDED_LEGAL`, and `resolveEffectiveCampaignLimit` returns 0 for that status. **FACT**: `SubscriptionService.java:856-887,744-746`.
10. `enterTermsPending` is called only from `TestSubscriptionController`. **FACT**: grep over `src/main`. So production probably has no `TERMS_PENDING` rows today. **INFERENCE**.

**Invoicing and data**

11. `findRetryable` selects all `PENDING`/`FAILED` invoices under the retry limit, for every company. **FACT**: `InvoiceRecordRepository.java:13-18`.
12. The April migration didn't touch `invoice_record`. **FACT**: `09-04-2026-downgrade-all-to-free-for-rollout.sql`. That old retryable invoices exist in production is a **HYPOTHESIS**; check it with a count query before the flip.
13. Plan limits in the database are FREE = 5 (after migration), BUSINESS = 5, ENTERPRISE = 10. The frontend plan list says 2 / 5 / 10. **FACT**: `22-03-2026-subscription-tables.sql:192-195`; `09-04-2026-update-free-plan-campaign-limit.sql:8`; `plan-billing.component.ts:72-106`.
14. Stripe price IDs are fixed in yml and the seed. **FACT**: `application.yml:638-640`. Whether they are live-mode prices is a **HYPOTHESIS** to check.
15. `getStatus` creates a subscription row as a side effect (`getOrCreateSubscription`). **FACT**: `SubscriptionService.java:103-104,708-711`.

**Frontend**

16. The public-config service assumes `true` if the fetch fails. Plan-billing's `paymentsEnabled` starts as `true` and controls the trial offer, cancel-trial, portal and plan cards, but not the payment-failed or pending-downgrade banners. **FACT**: `public-config.service.ts:21-25`; `plan-billing.component.ts:151-153`; `plan-billing.component.html:143,172,215,264,275,302,355`.
17. The layout nudge checks `/public-config` before calling `/status`. **FACT**: `layout.component.ts:254-268`. The spec asserts this. **FACT**: `layout.component.spec.ts:239-261`.
18. The frontend status DTO is hand-kept in `api-frozen/hidden-models`, because the backend controllers are `@Hidden`. **FACT**: `subscription.service.ts:3-10`; `SubscriptionController.java:31`.

**Tests**

19. Tests pin which beans exist when on vs off, and the "off" end-to-end suite runs in its own Failsafe run. **FACT**: `PaymentsTogglePresence_IntegrationTest.java:35-55`; `pom.xml:1912-1939`.
20. A PILOT-mode Spring context can share the "on" test contexts only if `mode` defaults to `ALL`. **INFERENCE** from 19 and the test property files setting the flag to `true`.
21. Spring can't load or unload a component per company. That's why the per-company decision has to be a runtime check and not an `@Conditional`. **INFERENCE** (standard Spring behaviour).
