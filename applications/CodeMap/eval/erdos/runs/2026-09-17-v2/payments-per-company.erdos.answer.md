I've mapped the switch in both repositories. The main finding: today one global setting, `app.payments.enabled`, both loads the payment machinery and decides who may use it. The pilot needs those two jobs separated.

## Problem
Paid subscriptions (Stripe checkout, trial, downgrade, portal, Fakturownia invoices) need to work for a named set of pilot companies. Every other company must keep today's free-only experience. That means no paid buttons, no trial, no Stripe calls and no change to their campaign limit or subscription state. Done means: the pilot companies can run trial → upgrade → payment → invoice end to end, other companies see no difference, and switching the pilot off is safe.

## Where it lives today
Most of the backend is in subsystem [11] (Subscriptions, payments & consent), package `backend/src/main/java/com/sm/instagram/platform/subscription/`. The frontend is in [174] (plan, billing & consent), plus the layout in [177].

**The switch itself**
- `config/AppPaymentsProperties.java`: one boolean, `app.payments.enabled`, which defaults to `true` in Java.
- `backend/src/main/resources/application.yml:410-411` sets it to `${APP_PAYMENTS_ENABLED:false}`.
- The test properties (`application.properties`, `-test`, `-integration`, `-e2e`) set it to `true`. `pom.xml:1913-1935` runs a separate Failsafe fork with it set to `false`.

**Startup**
- `config/PaymentsDisabledBootGuard.java` only loads when the switch is `false`. It refuses to start if any row is in TRIAL_ENTERPRISE, BUSINESS_ACTIVE, ENTERPRISE_ACTIVE, DOWNGRADE_PENDING, PAYMENT_FAILED or TERMS_PENDING.
- `stripe/StripeConfig.java` sets `Stripe.apiKey` only when the switch is on.
- `stripe/StripePropertiesConfig.java` always loads, so `StripeService` can still be wired while payments are off.
- `db/changelog/2026/04/09-04-2026-downgrade-all-to-free-for-rollout.sql` reset every row to FREE_ACTIVE, cleared Stripe IDs, and says no paying users exist.

**Endpoints**
- `SubscriptionPaidController.java` (`/subscription/trial/activate`, `/consent`, `/upgrade`, `/downgrade`, `/downgrade/cancel`, `/portal`, `/config`) exists only when the switch is on.
- `SubscriptionController.java` (`GET /status`, `GET /invoices`) is always on.
- `common/publicconfig/PublicConfigController.java` returns `PublicConfigDto(paymentsEnabled)` anonymously with `no-store`.

**Webhooks**
- `stripe/StripeWebhookController.java` exists only when the switch is on. Security already lets the path through (`WebSecurityConfiguration.java:154`, `JwtAuthenticationFilter.java:552`).
- It hands events to `stripe/StripeWebhookHandler.java`, which has three idempotency layers. That calls `SubscriptionService.handle*`.
- `StripeDevEventPoller.java` only runs under the `dev-poller` profile.

**Service guard**
- `SubscriptionService.requirePaymentsEnabled()` (lines 45-49) throws `PaymentsDisabledException`. It is called in `activateTrial`, `initiateUpgrade`, all five webhook handlers, `requestDowngrade` and `cancelDowngrade`.
- It is **not** called in `getStatus`, the cron processors, `enterTermsPending`, `processExpiredGracePeriods` or `deactivateForAccountDeletion`.

**Scheduled jobs**
- `cron/InvoiceRetryCronJob.java` and `cron/TrialExpiryNotifierCronJob.java` exist only when the switch is on.
- `cron/SubscriptionPeriodProcessorCronJob.java` checks the switch inside: trial and downgrade expiry run only when on, while free billing-period renewal always runs.
- `cron/TermsGraceProcessorCronJob.java` does nothing when the switch is off.

**Invoicing**
- `event/InvoiceCreatedEventListener.java` exists only when the switch is on. After the transaction commits it sends the invoice to Fakturownia; `InvoiceRetryCronJob` is the safety net.
- Invoice records are created only in `handleInvoicePaid`, and only when the amount is above zero.
- `fakturownia.api-key` defaults to blank, and the config comment says a blank key queues invoices for retry.

**Campaign limit**
- `CampaignLimitService.enforceLimit` → `resolveEffectiveCampaignLimit` reads the limit from the plan and status. `PartnershipOpportunityService.java:807` calls it.

**Frontend**
- `frontend/src/app/core/config/public-config.service.ts` caches `paymentsEnabled` and **falls back to `true` if the fetch fails**.
- `feature/plan-billing/plan-billing.component.ts:151` / `.html` use that flag to show the trial box, the "payments disabled" note, cancel-trial, the portal row and the plan cards.
- `layout/layout.component.ts:254-268` only calls `/subscription/status` (for the trial nudge) when the flag is true.
- `core/subscription/subscription.service.ts` and `core/api-frozen/subscription.client.ts` / `hidden-models.ts` are hand-maintained because the controllers are hidden from OpenAPI. `api/model/public-config-dto.ts` is generated.

**Tests pinned to the global split**
- Backend: `PaymentsTogglePresence_IntegrationTest`, `SubscriptionService_PaymentsDisabled_IntegrationTest`, `SubscriptionService_PaymentsToggleUnitTest`, `PaymentsDisabledBootGuardUnitTest`, `PublicConfigController{Unit,_Integration}Test`, `features/payments_off/payments-off.feature` + `RunPaymentsDisabledIT`, plus the subscription unit, integration and e2e suites.
- Frontend: `plan-billing.component.spec.ts`, `layout.component.spec.ts`, `sandbox/fixtures/plan-billing.fixture.ts`, `core/demo/demo-fixtures.ts:1619`.
- Frontend e2e (not in the graph): `e2e-tests/integration/flows/payments-off.spec.ts`, `subscription-lifecycle.spec.ts`, `e2e-tests/bdd/features/payments-off.feature`.

**Main flows**
- Upgrade: FE → `POST /subscription/upgrade` → `initiateUpgrade` → Stripe Checkout.
- Stripe → `/webhooks/stripe` → `handleCheckoutCompleted` sets BUSINESS or ENTERPRISE status → `handleInvoicePaid` creates an `InvoiceRecord` → `InvoiceCreatedEvent` → Fakturownia.
- The crons move trials, downgrades and grace periods forward.

A company is a `User` with the COMPANY authority. `company_subscription.user_id` is unique, so the pilot set is keyed by user id.

## Proposed change
Split the one switch into two ideas:

- **Infrastructure on or off** stays `app.payments.enabled`, with the same bean gating. This keeps the existing on/off split and its tests.
- **Who may use payments** is a new setting, `app.payments.audience = ALL | PILOT`. It defaults to `ALL`, so every current `enabled=true` environment and test behaves exactly as today. Production uses `enabled=true, audience=PILOT`.

The pilot list goes in a **database table**, `payments_pilot_company(user_id PK FK, added_by, added_time, note)`, added by a Liquibase changeset. Membership is read by a new `PaymentsEligibilityService`.

- **Chosen: database table.** Membership changes without a redeploy and leaves a record of who was added.
- **Rejected: property list** (`app.payments.pilot-user-ids`). It is simpler, but every change needs a restart and there is no record. It is acceptable only for a very small pilot that never changes.

`PaymentsEligibilityService` answers two questions, and keeping them separate protects anyone already paying:
1. `canStartPaidFlow(userId)` = infrastructure is on AND (audience is ALL OR the company is on the list). This gates trial, consent, upgrade and the paid buttons in the UI.
2. `canManageExistingPaid(userId)` = `canStartPaidFlow` OR the row already has a Stripe customer ID or a status other than FREE_ACTIVE. This gates portal, downgrade and cancel-downgrade, so a company removed from the pilot can still cancel or manage what it already pays for.

Webhooks and invoicing stay gated **only** on infrastructure. Money events for an existing Stripe subscription must never be dropped because the list changed.

The frontend takes its decision from the logged-in company's own `/subscription/status`, via a new `paymentsEnabled` field on `SubscriptionStatusDtoOut`. It no longer relies on the anonymous public config. That removes the risk of the "fall back to `true`" rule showing paid buttons to non-pilot companies. `PublicConfigDto.paymentsEnabled` becomes `enabled && audience == ALL`, so the landing page advertises nothing during the pilot. A new `paymentsPilot` field tells the layout that a status call is worth making.

## Plan
Each step leaves production working with `enabled=false`.

1. **Schema.** Add the changeset `db/changelog/2026/09/…-payments-pilot-company.sql`: the table, an FK to the user table and a PK on `user_id`. Register it in `db/changelog/changelog.xml`. No data yet.
2. **Eligibility service.**
   - Add `audience` (enum, default `ALL`) to `AppPaymentsProperties.java`.
   - New `subscription/pilot/PaymentsPilotCompany.java` + `PaymentsPilotCompanyRepository.java`.
   - New `subscription/config/PaymentsEligibilityService.java` with the two checks above.
   - Unit test covering every combination of enabled, audience, on the list and existing paid state.
3. **Service guards** in `SubscriptionService.java`:
   - Replace `requirePaymentsEnabled()` with `requireCanStartPaidFlow(userId)` in `activateTrial` and `initiateUpgrade`, and with `requireCanManageExistingPaid(userId)` in `requestDowngrade` and `cancelDowngrade`.
   - Keep the infrastructure-only check in the five `handle*` webhook methods.
   - `isTrialEligible` also requires `canStartPaidFlow`.
   - `getStatus` fills the new `paymentsEnabled` field (`canStartPaidFlow`) and a `paymentsManageable` field (`canManageExistingPaid`). Update `dto/SubscriptionStatusDtoOut.java`.
4. **Controller.** In `SubscriptionPaidController.java`, `/consent` checks `canStartPaidFlow` (via `LegalConsentService.recordSubscriptionConsent` or a guard in the controller), and `/portal` checks `canManageExistingPaid`, because it calls `stripeService` directly and skips the service guard.
5. **Crons.**
   - `SubscriptionService.processExpiredGracePeriods` and `processTrialExpiryInTermsPending` only act on rows where `canManageExistingPaid` holds, or keep the no-op for the non-pilot population.
   - Reason: `enterTermsPending` moves **every** status, FREE included, into TERMS_PENDING. With the infrastructure on, `TermsGraceProcessorCronJob` would otherwise move non-pilot FREE companies to SUSPENDED_LEGAL, which gives them a campaign limit of 0.
   - `processExpiredTrials`, `processExpiredDowngrades`, `sendTrialEndingReminders` and `InvoiceRetryCronJob` only find rows that paid flows create, so they need no change.
6. **Startup check.**
   - New `config/PaymentsPilotBootGuard.java`, loaded when `enabled=true` and `audience=PILOT`. It fails if any row in the six in-flight paid statuses belongs to a company that fails `canManageExistingPaid`. That can only happen through hand edits.
   - It logs the size of the pilot list.
   - `PaymentsDisabledBootGuard` stays as it is; it is still the gate for turning payments back off.
7. **Public config.**
   - `PublicConfigDto` becomes `(paymentsEnabled, paymentsPilot)`; `PublicConfigController` computes both.
   - Regenerate the OpenAPI spec (`backend/docs/openapi/openapi.json`, `frontend/docs/openapi/openapi.json`) and the frontend client with `openapi:gen`, which updates `api/model/public-config-dto.ts`.
8. **Frontend**, deployed after the backend. The fields are additive, so an old frontend against the new backend still works.
   - Add `paymentsEnabled` / `paymentsManageable` to `core/api-frozen/hidden-models.ts` and `testing/contract/subscription.contract.ts`.
   - `plan-billing.component.ts/.html`: trial, upgrade and plan cards use `status.paymentsEnabled`; portal and cancel-trial use `status.paymentsManageable`. The "payments disabled" note shows when neither is true.
   - `layout.component.ts:254-268`: call `getStatus()` when `paymentsEnabled || paymentsPilot`, and show the trial nudge only when `sub.paymentsEnabled`.
   - `public-config.service.ts`: expose `paymentsPilot()`. Keep the fall-back to `true`, but it no longer decides paid buttons.
   - Update `plan-billing.fixture.ts`, `demo-fixtures.ts:1619`, and the plan-billing and layout specs.
9. **Tests.**
   - Backend: add a PILOT group to `PaymentsTogglePresence_IntegrationTest` (same beans as ON). Add `SubscriptionService_PaymentsPilot_IntegrationTest`: a listed company can trial and upgrade, an unlisted one gets `PaymentsDisabledException`, a removed but paying company can still downgrade and use the portal, webhooks still process for a removed company, and the grace cron leaves unlisted FREE rows alone.
   - Extend `PublicConfigController_IntegrationTest`. Add `@payments-pilot` scenarios to a feature file with a runner forked like `RunPaymentsDisabledIT`, or use `@TestPropertySource`.
   - Frontend: a pilot/non-pilot spec in `e2e-tests/integration/flows/`.
10. **Operations** (`docs/ROLLOUT.md` Stage 5):
    - Set `STRIPE_PRIVATE_KEY`, `STRIPE_PUBLIC_KEY`, `STRIPE_WEBHOOK_SECRET`, `FAKTUROWNIA_API_KEY` and `FAKTUROWNIA_DOC_KEY`, and register the Stripe webhook endpoint.
    - Deploy with `APP_PAYMENTS_ENABLED=true`, `APP_PAYMENTS_AUDIENCE=PILOT` and an **empty** list, and confirm nothing changed.
    - Add the pilot companies. Run the flow on a test key, then on the live key.
    - Rollback: remove companies from the list (existing subscribers stay manageable). A full switch-off needs the Stripe subscriptions cancelled and the rows reconciled first, otherwise `PaymentsDisabledBootGuard` blocks startup by design. The April changeset has already run and will not run again.

## Risks and invariants
- **Non-pilot companies see no change**: no paid buttons, no trial, no Stripe customer, FREE limit unchanged. Guarded by the new pilot integration test, the frontend plan-billing spec, and a pilot e2e flow.
- **Money events are never dropped**: webhooks and invoicing depend only on the infrastructure setting, never on the list. The three idempotency layers in `StripeWebhookHandler` (event-id check, unique constraint, optimistic lock → 503) stay unchanged. Guarded by `SubscriptionService_Webhook_IntegrationTest`, `StripeWebhookHandlerUnitTest` and the new "removed but paying" case.
- **No stranded rows**: a paid status must always be resolvable by a cron or a management action. The two separate checks plus `PaymentsPilotBootGuard` enforce this; `PaymentsDisabledBootGuardUnitTest` still covers switching off.
- **Terms and grace processing**: switching the infrastructure on activates `TermsGraceProcessorCronJob` for everyone. Without step 5, a terms bump would suspend free companies. Guarded by a new cron integration test and `SubscriptionService_Cron_IntegrationTest`.
- **Existing environments stay as they are**: `audience` defaults to `ALL`. Guarded by the existing on/off `PaymentsTogglePresence_IntegrationTest` and `payments-off.feature`, which must pass unchanged.
- **Consent (EU Art. 16(m))**: consent must still be recorded before activation, and only for eligible companies. Guarded by `SubscriptionService_Terms_IntegrationTest` and the plan-billing spec.
- **Frontend fallback**: a failed `/public-config` call must not expose paid buttons. Handled by the per-company status field; add a spec where `getConfig` errors.
- **Stripe key is global**: `Stripe.apiKey` is a static value set at startup, so live mode applies to the whole process. That is harmless for non-pilot companies because no code path calls Stripe for them. Watch for test-key/live-key mix-ups between environments.
- **Company vs team members**: eligibility is keyed by `user_id`. If company team members act on behalf of another user id, the check must resolve to the owner (see the hypothesis below).

## Evidence
- FACT: `AppPaymentsProperties` has one boolean `enabled`, defaulting to `true` (`subscription/config/AppPaymentsProperties.java:23-27`). `application.yml:410-411` binds it to `${APP_PAYMENTS_ENABLED:false}`.
- FACT: `PaymentsDisabledBootGuard` loads when the switch is `false` and throws if `countByStatusIn` finds any of six in-flight paid statuses (`config/PaymentsDisabledBootGuard.java:24,35-55`).
- FACT: `SubscriptionPaidController`, `StripeWebhookController`, `StripeConfig`, `InvoiceRetryCronJob`, `TrialExpiryNotifierCronJob` and `InvoiceCreatedEventListener` all carry `@ConditionalOnProperty(app.payments.enabled=true, matchIfMissing=false)` (lines read in each file). `PaymentsTogglePresence_IntegrationTest.java:35-55` pins this split.
- FACT: `SubscriptionPeriodProcessorCronJob` runs `processExpiredTrials`/`processExpiredDowngrades` only when the switch is on, and `renewExpiredFreeBillingPeriods` always (`cron/SubscriptionPeriodProcessorCronJob.java:36-41`). `TermsGraceProcessorCronJob` returns early when off (line 37).
- FACT: `requirePaymentsEnabled()` is called at `SubscriptionService.java` lines 61, 137, 208, 261, 317, 341, 381, 443 and 513, which are `activateTrial`, `initiateUpgrade`, the five webhook handlers, `requestDowngrade` and `cancelDowngrade`. It is not called in `getStatus`, `enterTermsPending` or `processExpiredGracePeriods`.
- FACT: `enterTermsPending` moves FREE_ACTIVE and every paid status into TERMS_PENDING (`SubscriptionService.java:800-829`). `processExpiredGracePeriods` sets SUSPENDED_LEGAL (873). `resolveEffectiveCampaignLimit` returns 0 for SUSPENDED_LEGAL (744-746). The only caller of `enterTermsPending` in main code is `TestSubscriptionController.java:379` (grep of `backend/src/main`).
- INFERENCE: once the infrastructure is on, a future production path that calls `enterTermsPending` would suspend non-pilot free companies after 38 days. It follows from the three facts above; today no production caller exists, so this is a latent risk rather than a current bug.
- FACT: `handleInvoicePaid` is the only place in `SubscriptionService` that creates an `InvoiceRecord` and publishes `InvoiceCreatedEvent`, and only when `amountPaid > 0` (lines 299-312). The listener sends to Fakturownia after commit and leaves failures for the cron (`event/InvoiceCreatedEventListener.java:28-39`).
- FACT: `/portal` calls `stripeService.createPortalSession` directly with no service guard (`SubscriptionPaidController.java:95-107`).
- FACT: `CompanySubscription.user` is `unique = true` and there is a `@Version` field (`entity/CompanySubscription.java:33-39`). `CampaignLimitService.enforceLimit` uses `resolveEffectiveCampaignLimit` (lines 23-35). It is called from `PartnershipOpportunityService.java:807` (grep).
- FACT: `StripeWebhookHandler` implements the event-id check, unique-constraint catch and optimistic-lock rethrow (`stripe/StripeWebhookHandler.java:41-60`). The controller returns 503 on a lock conflict (`StripeWebhookController.java:51-58`).
- FACT: `StripeConfig` sets the static `Stripe.apiKey` (`stripe/StripeConfig.java:28-30`).
- FACT: `PublicConfigController` returns `PublicConfigDto(appPaymentsProperties.isEnabled())` with `no-store` (`common/publicconfig/PublicConfigController.java:30-35`).
- FACT: `public-config.service.ts` maps a missing value or an error to `true` (lines 21-25). `plan-billing.component.ts:151-153` uses it with `initialValue: true`, and the template gates trial, cancel-trial, portal and plan cards on it (`plan-billing.component.html:214-215, 264, 275, 302, 355`). `layout.component.ts:254-268` calls `getStatus()` only when it is true.
- FACT: `SubscriptionController` and `SubscriptionPaidController` are `@Hidden`. The frontend imports their types from `core/api-frozen/hidden-models.ts` (`subscription.service.ts:3-10`), so those types are maintained by hand, not generated.
- FACT: the April changeset reset every non-FREE row and cleared Stripe IDs, and its rollback is a no-op (`09-04-2026-downgrade-all-to-free-for-rollout.sql:35-48, 82-84`).
- FACT: `ROLLOUT.md` Stage 5 lists the Stripe and Fakturownia variables and says the boot guard must not be bypassed (`backend/docs/ROLLOUT.md:113-131`).
- FACT: `deactivateForAccountDeletion` cancels Stripe resources when IDs are present, and Stripe failures do not block deletion (`SubscriptionService.java:898-923`). It is called from `UserAccountOrchestrator.java:64` (grep).
- INFERENCE: `processExpiredTrials`, `processExpiredDowngrades`, `sendTrialEndingReminders` and invoice retry only touch rows created by guarded paid flows. Their repository queries filter on paid statuses or pending invoices (`CompanySubscriptionRepository.java:23-39`), so non-pilot companies cannot reach them once step 3 is in place.
- HYPOTHESIS: `InvoiceRetryService` and `FakturowniaAdapter` treat a blank API key as "not configured" and leave invoices PENDING or FAILED. This comes from the `application.yml:647-650` comment, not the adapter code. Check by reading `invoicing/FakturowniaAdapter.java` and `InvoiceRetryService.java`.
- HYPOTHESIS: company team members act under their own `user_id`, not the owner's. The graph shows no backend Team or Company entity, and the frontend has a `team` screen. Check by grepping the backend for the frontend team service's HTTP paths before keying eligibility on the logged-in user.
- HYPOTHESIS: nothing on the landing or marketing pages depends on `paymentsEnabled`. The grep under `frontend/src/app` found only the layout, plan-billing, fixtures and generated files; pricing on the landing page may be static copy. Check `feature/landing` if the pilot must not advertise prices.
- HYPOTHESIS: `db/changelog/changelog.xml` lists changesets explicitly rather than including folders. Check before adding the step 1 changeset.
