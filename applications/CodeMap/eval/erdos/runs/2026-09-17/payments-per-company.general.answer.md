## Problem

Payments are one switch for the whole deployment: `app.payments.enabled`, set from `APP_PAYMENTS_ENABLED` and defaulting to `false`. It decides which Spring beans exist when the app starts. Nothing in the code can tell companies apart. The goal is to turn on paid subscriptions (trial, upgrade, downgrade, portal, Stripe webhooks, Fakturownia invoices) for a named pilot group, while every other company keeps today's free-only behaviour and screens.

A "company" here is a `User` with authority `COMPANY`. It has exactly one `company_subscription` row, keyed by `user_id` (unique). There is no separate company or organisation entity.

## Where it lives today

**Backend: the switch and its gates** (`backend/src/main/java/com/sm/instagram/platform/...`)

| Concern | File | How it is gated |
|---|---|---|
| Property | `backend/src/main/resources/application.yml:410-411`, `subscription/config/AppPaymentsProperties.java` (Java default `true`), `AppPaymentsConfiguration.java` | – |
| Startup | `subscription/config/PaymentsDisabledBootGuard.java` | Only loads when the switch is `false`. Stops startup if any row is `TRIAL_ENTERPRISE`, `BUSINESS_ACTIVE`, `ENTERPRISE_ACTIVE`, `DOWNGRADE_PENDING`, `PAYMENT_FAILED` or `TERMS_PENDING`. |
| Startup data | `db/changelog/2026/04/09-04-2026-downgrade-all-to-free-for-rollout.sql` (listed at `changelog.xml:175-176`) | Reset every row to `FREE_ACTIVE` and cleared all Stripe IDs. |
| Stripe SDK | `subscription/stripe/StripeConfig.java` (sets the global `Stripe.apiKey`) | Bean only exists when the switch is `true`. `StripePropertiesConfig` and `StripeService` always exist. |
| Paid endpoints | `subscription/SubscriptionPaidController.java` (trial, consent, upgrade, downgrade, downgrade cancel, portal, config) | Bean only exists when `true`. The portal and consent endpoints call no service-level check. |
| Read endpoints | `subscription/SubscriptionController.java` (`/status`, `/invoices`) | Always on |
| Webhook | `subscription/stripe/StripeWebhookController.java` → `StripeWebhookHandler.java` | Bean only exists when `true`. Allowed without login in `WebSecurityConfiguration.java:154` and `JwtAuthenticationFilter.java:552`. |
| Service check | `subscription/SubscriptionService.java:45-49` `requirePaymentsEnabled()` | Called by activateTrial, initiateUpgrade, the 5 webhook handlers, requestDowngrade and cancelDowngrade. It throws `PaymentsDisabledException`, which becomes HTTP 503 (`BusinessExceptionHandler.java:121`). |
| Scheduled jobs | `cron/SubscriptionPeriodProcessorCronJob.java` | Always on. The trial and downgrade expiry steps only run when the switch is on; free billing-period renewal always runs. |
| | `cron/TermsGraceProcessorCronJob.java` | Always on; does nothing when the switch is off |
| | `cron/TrialExpiryNotifierCronJob.java`, `cron/InvoiceRetryCronJob.java` | Beans only exist when `true` |
| Invoicing | `event/InvoiceCreatedEventListener.java` | Bean only exists when `true` |
| | `invoicing/InvoiceRetryService.java`, `FakturowniaAdapter.java`, `FakturowniaConfig.java` | Always loaded; the adapter has its own `fakturownia.enabled` switch |
| Public flag | `common/publicconfig/PublicConfigController.java`, `PublicConfigDto(boolean paymentsEnabled)` | Anonymous, `no-store` |
| Test-only | `subscription/TestSubscriptionController.java` (profiles `e2e`/`dev`) | Calls the service directly |

**Flows that matter**
- **Entry into paid:** `initiateUpgrade` is the only place a Stripe customer is created. After that, Stripe sends `checkout.session.completed`, which `handleCheckoutCompleted` looks up by `stripeCustomerId`.
- **Renewal and invoicing:** `invoice.paid` → `handleInvoicePaid` creates an `InvoiceRecord` and publishes `InvoiceCreatedEvent`. The listener sends it to Fakturownia after commit, and the retry job resends failures every 15 minutes.
- **Webhook routing:** all events arrive on one account-wide endpoint and are matched to a company only by Stripe customer or subscription ID.
- **Status:** `getStatus` returns `trialEligible=true` for every free company that never had a Stripe customer, whatever the switch says.

**Frontend** (`frontend/src/app/...`)
- `core/config/public-config.service.ts`: reads `paymentsEnabled` and caches it. If the flag is missing or the call fails, it treats payments as on (`?? true`, `catchError → true`).
- `feature/plan-billing/plan-billing.component.ts:151` (`toSignal(..., initialValue: true)`) and `.html:215, 264, 275, 302, 355`: the trial prompt, the "payments disabled" notice, trial cancel, portal and the plans list are all gated on that one flag.
- `layout/layout.component.ts:254-268`: the trial nudge. If public-config says off, it never calls `/subscription/status`.
- `core/subscription/subscription.service.ts`: `SubscriptionWriteApi` assumes the endpoints exist.
- `core/api-frozen/hidden-models.ts:43` is hand-written, because the controllers are `@Hidden` from OpenAPI. `api/model/public-config-dto.ts` is generated.
- Places that embed the flag: `core/demo/demo-fixtures.ts:1619`, `sandbox/fixtures/plan-billing.fixture.ts:164-174`, `app.routes.ts:563, 597-606` (the Stripe return URLs redirect to plan-billing).

**Tests that guard the switch today**
- `PaymentsTogglePresence_IntegrationTest` (which beans exist when on vs off)
- `PaymentsDisabledBootGuardUnitTest`
- `SubscriptionService_PaymentsToggleUnitTest`
- `SubscriptionService_PaymentsDisabled_IntegrationTest`
- `PublicConfigController_IntegrationTest` / `PublicConfigControllerUnitTest`
- `features/payments_off/payments-off.feature` run by `RunPaymentsDisabledIT`, in its own Failsafe fork (`pom.xml:1917-1939`)
- Test property files that set `app.payments.enabled=true`: `application.properties:92`, `-test:198`, `-integration:225`, `-e2e:43`
- Frontend: `layout.component.spec.ts`, `plan-billing.component.spec.ts`, `demo-fixtures.account.spec.ts:174`, and the e2e framework `payments-config.api.ts` / `subscription.api.ts`

## Proposed change

**1. Split "is the plumbing on" from "who may use it".**
- Keep `app.payments.enabled` as the **infrastructure** switch, unchanged. Every `@ConditionalOnProperty`, the existing boot guard and the payments-off suite stay valid.
- Add `app.payments.audience: PILOT | ALL` (`${APP_PAYMENTS_AUDIENCE:PILOT}`), read only when infrastructure is on. The default is **PILOT**, so a forgotten variable fails closed: turning infrastructure on alone never opens payments to everyone.
- Why not gate per company at the bean or webhook level? Stripe sends every event to one account-wide endpoint, and events can only be tied to a company after a database lookup. So the plumbing has to be on for the whole deployment, and the per-company decision has to live in the service.

**2. Store pilot membership in a new table, not in config and not on `company_subscription`.**
- Table: `payments_pilot_company(user_id PK/FK, added_at, added_by, entry_allowed boolean)`.
- Not on `company_subscription`: that row is created lazily, was bulk-rewritten by the April migration, and is deleted by the test reset endpoint.
- Not an ID list in config: IDs differ per environment and there would be no audit trail.
- Membership is added or paused by a reviewed Liquibase changeset or runbook SQL. Rows are **never deleted**; pausing a company means `entry_allowed=false`.

**3. Add one policy bean, `PaymentsAccessPolicy`, with two questions:**
- `canEnterPaid(userId)` = infrastructure on AND (audience ALL OR a pilot row with `entry_allowed`). Used for trial, consent, and upgrade from free.
- `canManagePaid(userId)` = infrastructure on AND (audience ALL OR any pilot row OR the subscription already has a Stripe customer). Used for downgrade, cancel-downgrade, portal, and upgrade of an existing Stripe subscription.
- Webhook handlers and scheduled jobs keep the **global** infrastructure check only. Events for an existing Stripe subscription are always processed, so a paused pilot company is never left with money taken and no database update.
- A non-pilot company that calls an entry endpoint gets a new `PaymentsNotAvailableException`, which becomes **403**. That keeps it distinct from the 503 "switched off" case.

**4. Frontend contract**
- `PublicConfigDto` gains `paymentsAudience: NONE | PILOT | ALL`.
- `paymentsEnabled` stays and means `enabled && audience==ALL`. Anonymous pages and older clients therefore see "off" while in PILOT mode.
- `SubscriptionStatusDtoOut` gains `paidEntryAvailable` and `paidManagementAvailable`. In PILOT mode, `trialEligible` is also forced to false when `canEnterPaid` is false.
- The frontend gates paid actions on these status fields, which **fail closed**: the status call is the same one that loads the page, so a failed call shows the page's error state, not "hidden billing".
- The layout calls `/subscription/status` only when `paymentsAudience != NONE`. An OFF deployment keeps making no status call.

**5. A second startup guard for PILOT mode**
- `PaymentsPilotBootGuard` loads when `enabled=true && audience=PILOT`.
- It stops startup if any company without a pilot row is in a paid in-flight status or has `stripe_customer_id` set.
- This makes the isolation invariant fail fast, the same way the existing guard does.

## Plan

1. **Migration.** Add `db/changelog/2026/09/<date>-payments-pilot-company.sql` (table plus FK to users) and include it in `changelog.xml`. Seed no rows.
2. **Properties and policy.** Add `audience` to `AppPaymentsProperties.java` and `application.yml:410` (`audience: ${APP_PAYMENTS_AUDIENCE:PILOT}`). New files: `subscription/config/PaymentsAudience.java`, `subscription/config/PaymentsAccessPolicy.java`, `subscription/entity/PaymentsPilotCompany.java`, `subscription/repository/PaymentsPilotCompanyRepository.java`, and `subscription/exception/PaymentsNotAvailableException.java` plus a 403 handler in `BusinessExceptionHandler.java`.
3. **Service checks** in `SubscriptionService.java`:
   - `activateTrial` → `requireEntry(userId)`
   - `initiateUpgrade` → entry check, or manage check when `stripeSubscriptionId != null`
   - `requestDowngrade` and `cancelDowngrade` → `requireManage(userId)`
   - The 5 webhook handlers keep `requirePaymentsEnabled()`
   - `getStatus` fills the two new fields and masks `trialEligible`
   - Add `paidEntryAvailable` and `paidManagementAvailable` to `dto/SubscriptionStatusDtoOut.java`
4. **Controller gaps.** In `SubscriptionPaidController.java`, add `requireManage` to `/portal` and `requireEntry` to `/consent`. Neither calls a service check today.
5. **Public config.** `PublicConfigDto.java` and `PublicConfigController.java`: add `paymentsAudience` and redefine `paymentsEnabled`.
6. **Startup guard.** New `subscription/config/PaymentsPilotBootGuard.java`, plus a repository query on `CompanySubscriptionRepository.java` that counts paid or Stripe-linked rows with no pilot row.
7. **Keep the existing suites green.** Add `app.payments.audience=ALL` next to `app.payments.enabled=true` in the four test property files.
8. **New backend tests:**
   - `PaymentsAccessPolicyUnitTest` (the full matrix)
   - `SubscriptionService_PaymentsPilot_IntegrationTest`: a non-pilot company gets 403 on trial, upgrade, consent and portal; a pilot company succeeds; a paused pilot company can still downgrade and use the portal; simulated webhooks still process for a paused company; a non-pilot free company is untouched by the period and grace jobs
   - `PaymentsPilotBootGuardUnitTest`
   - Extend `PaymentsTogglePresence_IntegrationTest` with a PILOT context: gated beans present, the old guard absent, the pilot guard present
   - Extend `PublicConfigController_IntegrationTest` with the PILOT case
   - Optional: a `payments-pilot.feature` suite with its own Failsafe execution in `pom.xml`
9. **Frontend API layer.** Regenerate `api/model/public-config-dto.ts` and hand-edit `core/api-frozen/hidden-models.ts` (the new status fields). Add `paymentsAudience()` to `core/config/public-config.service.ts`, keeping the fail-to-true behaviour only for the legacy `paymentsEnabled()`.
10. **Frontend screens.**
    - `feature/plan-billing/plan-billing.component.ts/.html`: entry actions (trial prompt, plans list upgrade buttons) use `status.paidEntryAvailable`; management actions (trial cancel, downgrade buttons, pending-downgrade cancel, portal) use `status.paidManagementAvailable`; the "payments disabled" notice shows when neither is true.
    - `layout/layout.component.ts:254`: gate on `paymentsAudience() !== 'NONE'` and `sub.paidEntryAvailable`.
11. **Frontend tests and fixtures.** Update `plan-billing.component.spec.ts`, `layout.component.spec.ts`, `sandbox/fixtures/plan-billing.fixture.ts` (add a pilot and a non-pilot fixture), `core/demo/demo-fixtures.ts` and its spec, and the e2e `payments-config.api.ts` / `subscription.api.ts`.
12. **Rollout and docs.** Update `backend/docs/ROLLOUT.md` (Stage 5): deploy with `enabled=false` (no change); run the migration; insert pilot rows; set `APP_PAYMENTS_ENABLED=true` and `APP_PAYMENTS_AUDIENCE=PILOT` along with the Stripe and Fakturownia secrets; verify on a Stripe test key first.
    - **Rollback:** set `APP_PAYMENTS_ENABLED=false`. The existing startup guard will then refuse to start while pilot rows are in paid states, which is intended; those companies must be reconciled or cancelled in Stripe first.

## Risks and invariants

| # | Invariant | Guarded by |
|---|---|---|
| I1 | With `enabled=false`, behaviour is byte-for-byte today's (same beans, guard, 404 webhook, `paymentsEnabled=false`) | Existing `PaymentsTogglePresence_IntegrationTest` (OFF), `payments-off.feature`, `PaymentsDisabledBootGuardUnitTest`, left unchanged |
| I2 | A non-pilot company can never create a Stripe customer, start a trial, or reach a paid status | New pilot integration test, `PaymentsPilotBootGuard` and its unit test |
| I3 | Every Stripe-linked subscription's webhooks are processed while infrastructure is on, whatever its pilot status | New integration test (paused pilot company plus simulated webhook), existing webhook handler tests |
| I4 | Free billing-period renewal keeps running for everyone in every mode | Existing `SubscriptionPeriodProcessorCronJob` path, plus a new assertion in the pilot integration test |
| I5 | Idempotency (Stripe event ID unique, optimistic lock → 503 retry) and the Fakturownia `oid` key are unchanged | Existing handler and invoice tests; this plan does not touch those files |
| I6 | Non-pilot and anonymous frontends show exactly today's OFF screens (no paid prompts, no trial nudge) | Updated `plan-billing.component.spec.ts` and `layout.component.spec.ts` for the NONE, PILOT-non-member and PILOT-member cases |
| I7 | A forgotten `APP_PAYMENTS_AUDIENCE` does not open payments to everyone | `PILOT` default; unit test on the property binding |

**Risks**
- With infrastructure on, `TermsGraceProcessorCronJob` starts running for all companies and can set `SUSPENDED_LEGAL` (limit 0) on any `TERMS_PENDING` row. This is safe today only because nothing in production code creates `TERMS_PENDING` (only `TestSubscriptionController` does) and the existing guard required zero such rows. I recommend limiting that job to pilot companies in PILOT mode.
- The frontend currently fails *open*. Leaving any paid action on `paymentsEnabled()` would show it to non-pilot companies during a config blip, where they would get 403s.
- `Stripe.apiKey` is a global static, so the Stripe SDK is initialised for the whole deployment once infrastructure is on. This is acceptable because `StripeService` is only called on entry or manage paths and webhooks.
- `StripeDevEventPoller` (profile `dev-poller`) is not gated by the switch; it is irrelevant in production.

## Evidence

**Read in the code (FACT)**
- F1: The switch defaults to `false` in `application.yml:411`; the Java default is `true` (`AppPaymentsProperties.java:27`).
- F2: Beans gated on the switch being `true`: `SubscriptionPaidController:47`, `StripeWebhookController:21`, `StripeConfig:23`, `TrialExpiryNotifierCronJob:15`, `InvoiceRetryCronJob:22`, `InvoiceCreatedEventListener:23`. The existing guard loads only on `false` (`PaymentsDisabledBootGuard:24`).
- F3: `requirePaymentsEnabled()` call sites are as listed. `/portal` and `/consent` in `SubscriptionPaidController` call no check (lines 67-72, 95-107).
- F4: The webhook matches events to companies by Stripe customer or subscription ID (`SubscriptionService:209, 265, 318, 342, 382`).
- F5: `getStatus` computes `trialEligible` with no reference to the switch (`SubscriptionService:123, 750-754`).
- F6: `enterTermsPending` is only called from `TestSubscriptionController:379` (grep of `src/main`).
- F7: The frontend fails open (`public-config.service.ts:22-23`, `plan-billing.component.ts:151-153`), and the layout skips the status call when the flag is off (`layout.component.ts:255-256`).
- F8: The subscription is unique per `user_id` (`CompanySubscription.java:38`); no organisation or team entity exists (grep).
- F9: Four test property files set the switch to `true`; the payments-off suite runs in its own fork (`pom.xml:1917-1939`).
- F10: `PaymentsDisabledException` → 503 (`BusinessExceptionHandler:117-126`).
- F11: `StripeService` sets no API key per request (no `apiKey`/`RequestOptions` found); it relies on `StripeConfig` setting the global key.

**Not yet confirmed (INFERENCE / HYPOTHESIS)**
- I-A (INFERENCE): `initiateUpgrade` is the only production code that sets `stripeCustomerId`, so I2 can be enforced at entry. Source: reading `SubscriptionService`; not yet grepped repo-wide.
- I-B (INFERENCE): `InvoiceRecord`s are only created in `handleInvoicePaid`, so invoicing is scoped automatically to Stripe-linked companies.
- I-C (INFERENCE): `layout.component.ts` and `plan-billing.component.ts` are the only production frontend consumers of the payments flag. Source: frontend grep excluding showcases.
- H-A (HYPOTHESIS): BUSINESS's campaign limit may not exceed FREE's, now raised to 5 (`09-04-2026-update-free-plan-campaign-limit.sql`). If so, the pilot BUSINESS plan offers no benefit; this is a product risk to check before the pilot.
- H-B (HYPOTHESIS): nothing in production code calls `acceptTerms` or `processExpiredGracePeriods` except the scheduled jobs and the test controller.

## Corrections after verification

I checked every claim the plan depends on that I hadn't read directly. None of them breaks the design, but one turns a product risk into a blocker and adds a step.

1. **H-A is true, and it is a blocker for the pilot.** In `db/changelog/2026/03/22-03-2026-subscription-tables.sql:192-195` the plans are seeded as FREE 2, BUSINESS 5 (29 PLN) and ENTERPRISE 10. `09-04-2026-update-free-plan-campaign-limit.sql:8` then raised FREE to 5, and no later changeset changes BUSINESS. So a pilot company paying 29 PLN gets the same 5 campaigns it has for free. The frontend also hard-codes FREE as 2 (`plan-billing.component.ts:72-106`), which no longer matches the database.
   - **New Step 0 (before Step 1):** get a product decision. Either ship a changeset that raises the BUSINESS (and possibly ENTERPRISE) limit, or offer only ENTERPRISE in the pilot.
   - **Addition to Step 10:** fix the hard-coded plan limits in `plan-billing.component.ts` so they match the database.
2. **I-A is true.** In production code, only `SubscriptionService.java:158` (`initiateUpgrade`) sets `stripeCustomerId`. The other writers are `TestSubscriptionController` (e2e/dev profiles only) and the old migration, which clears it. So I2 can be enforced at the entry point, and the pilot startup guard's "Stripe-linked" check is sound.
3. **I-B is true.** `InvoiceRecord` is created only at `SubscriptionService.java:301`, inside `handleInvoicePaid`, so invoicing only ever touches Stripe-linked companies.
4. **I-C is true.** `paymentsEnabled()` is used in production only by `layout.component.ts`, `plan-billing.component.ts/.html` and `public-config.service.ts`. The other hits are specs and the sandbox fixture, which are already in Step 11. `demo-fixtures.ts` stubs the URL, not the service.
5. **H-B is true, with one small correction.** `processExpiredGracePeriods` and `processTrialExpiryInTermsPending` are called only by `TermsGraceProcessorCronJob` and the test controller. `acceptTerms` is called only by the test controller. Its javadoc (`SubscriptionService.java:832`) mentions a "LegalConsentService event bridge", but that bridge is not wired: `LegalConsentService` never calls it. This doesn't change the plan. It does confirm that nothing in production creates or clears `TERMS_PENDING`, so limiting the grace job to pilot companies (Risks) is optional hardening, not a requirement.
