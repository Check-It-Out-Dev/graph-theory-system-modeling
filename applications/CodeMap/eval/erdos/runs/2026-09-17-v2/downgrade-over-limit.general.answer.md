# What happens when a paid company downgrades or its payment fails, with more campaigns than Free allows

## Problem

**What happens today:** nothing touches the campaigns the company already has. The plan limit is only checked when a new campaign is created (`PartnershipOpportunityService.java:806-808`). It is not a cap on how many campaigns are live at once. It counts campaigns created inside the current billing period, including soft-deleted ones.

Every move to Free (trial expiry, deletion by Stripe, the cron) starts a brand-new billing period at "now". The usage counter therefore drops to 0. A company that created 10 campaigns yesterday on Enterprise can create a full Free allowance today, and all 10 old campaigns stay live.

On top of that, the rules are encoded differently in different places:
- **Plan limits disagree:**
  - Database: FREE=5, BUSINESS=5, ENTERPRISE=10.
  - Frontend plan cards: FREE=2.
  - Trial banner copy: "10 campaigns/month".
  - FREE and BUSINESS having the same limit means a Business→Free downgrade changes nothing.
- **Stale plan fields in state transitions leak the wrong limit.** For example, a trial that expires while new terms are pending leaves a Free account on the Enterprise plan for good.
- **Scheduled jobs:** the cron branch meant to finish a downgrade can never match anything.
- **Notifications and screens:** a voluntary downgrade is reported as "all payment attempts failed". The downgrade dialog says nothing about campaigns, and the frontend has no handling for the limit-reached error.

## Where it lives today

**Backend (`backend/src/main/java/com/sm/instagram/platform/`)**
- `subscription/SubscriptionService.java` holds the state machine:
  - `requestDowngrade` (442-509), `cancelDowngrade` (512-550).
  - Stripe webhook handlers: `handlePaymentFailed` (316), `handleInvoicePaid` recovery (275-284), `handleSubscriptionDeleted` (340-377), `handleSubscriptionUpdated` (380-435).
  - Cron processors: `processExpiredTrials` (557), `processExpiredDowngrades` (591), `processTrialExpiryInTermsPending` (625), `renewExpiredFreeBillingPeriods` (669).
  - Terms flow: `enterTermsPending` / `acceptTerms` (800-849).
  - `resolveEffectiveCampaignLimit` (734-748).
- `subscription/CampaignLimitService.java` does the check (`count >= limit`, then throws `CampaignLimitExceededException` with code `CAMPAIGN_LIMIT`).
- `subscription/repository/BillingPeriodRepository.java`: `countCampaignsInPeriod` (28-33) and `findExpiredPendingDowngrades` (24-26).
- `subscription/stripe/StripeWebhookHandler.java` routes Stripe events: `invoice.payment_failed`, `customer.subscription.deleted`, `customer.subscription.updated`.
- `subscription/cron/SubscriptionPeriodProcessorCronJob.java` runs daily at 04:00. The paid-plan branches only run when payments are enabled.
- `subscription/config/PaymentsDisabledBootGuard.java`.
- `partnershipopportunities/PartnershipOpportunity.java`: soft-delete flag `active`, plus `createdTime`.
- `notification/NotificationType.java` and `notification/email/NotificationEmailService.java`.
- Database migrations (`src/main/resources/db/changelog/`):
  - `2026/03/22-03-2026-subscription-tables.sql`: seeds limits 2/5/10.
  - `2026/04/09-04-2026-update-free-plan-campaign-limit.sql`: raises FREE to 5.
  - `2026/04/09-04-2026-downgrade-all-to-free-for-rollout.sql`.
  - `2026/03/23-03-2026-subscription-notification-translations.sql`.

**Frontend (`frontend/src/`)**
- `app/feature/plan-billing/plan-billing.component.ts`:
  - `PLAN_LADDER` (72-106) hardcodes limits 2/5/10.
  - `availableDowngrades` (205-219), `UPGRADE_PLANS` (hardcoded prices).
- `plan-billing.component.html`: usage ring "used this period", pending-downgrade banner, payment-failed banner.
- `downgrade-confirm-dialog.component.{ts,html}`: copy only, no impact information.
- `app/core/demo/demo-fixtures.ts:1014-1022`, `app/sandbox/fixtures/plan-billing.fixture.ts`.
- `assets/i18n/en.json` (and `pl.json`):
  - `plan_billing.plans.limit` reads "{{n}} campaigns a month".
  - Unused `subscription.campaign_limit.*` dialog keys (6576-6583).
  - Banner `trial_offer` promises "10 campaigns/month".

**Flows that matter**
1. **Voluntary downgrade to Free:**
   - `requestDowngrade` sets status `DOWNGRADE_PENDING`, sets `targetPlan`, and tells Stripe to cancel at period end.
   - At period end Stripe sends `customer.subscription.deleted`. `handleSubscriptionDeleted` sets `FREE_ACTIVE` and opens a new FREE period from now.
   - It sends the notification **PAYMENT_EXHAUSTED**.
2. **Enterprise → Business:** a Stripe schedule fires `subscription.updated`. `handleSubscriptionUpdated` opens a new period from now and sends DOWNGRADED.
3. **Payment failure:**
   - `handlePaymentFailed` sets `PAYMENT_FAILED`; the paid limit is kept while Stripe retries.
   - If a retry succeeds, `handleInvoicePaid` restores the plan.
   - If retries run out, Stripe sends `subscription.deleted`, which takes flow 1's path.

## Proposed change

**One policy, stated once and enforced once:**

1. **The quota limits creation; it does not switch off what exists.** The plan limit is the number of campaigns a company may *create* in a rolling 30-day window.
   - Existing campaigns are never deactivated, hidden or deleted because of a plan change or a payment failure. Turning them off would break applications and collaborations in progress, and there is no suspended state for campaigns.
   - After a downgrade, creation is blocked until the 30-day count falls below the new limit.
   - Soft-deleted campaigns still count, so deleting and recreating can't get around the limit (this matches today's query).
2. **The window doesn't depend on the billing period** (`createdTime > now − 1 month`). This removes the counter reset on every plan change, and the double-count at the exact moment a Free period rolls over. Billing periods stay for invoicing and display only.
3. **The effective limit comes from `currentPlan`, and only from there.** In every in-between state (`DOWNGRADE_PENDING`, `PAYMENT_FAILED`, `TERMS_PENDING`) `currentPlan` is still the paid plan.
   - `SUSPENDED_LEGAL` and `ACCOUNT_DEACTIVATED` get 0.
   - `previousPlan` goes back to being an audit field and stops affecting the limit.
4. **Payment failure keeps the paid limit** for as long as Stripe keeps retrying. When retries run out the account falls back to Free under rule 1.
   - Recovery restores the *state* the account was in before, not a status derived from the plan. A downgrade pending before the failure stays pending.
   - Downgrade requests are not accepted while in `PAYMENT_FAILED`; the portal is the way out, which matches what the frontend already offers.
5. **Stripe decides when a downgrade happens.** The cron only reconciles missed events: a subscription in `DOWNGRADE_PENDING` targeting Free with no live Stripe subscription after the period end plus 48 hours. It runs through the same completion method as the webhook.
6. **The backend is the only source of plan limits and prices.** A public endpoint, `GET /subscription/plans`, serves them and the frontend renders it. The numbers themselves (FREE=5 equal to BUSINESS=5) are a **product decision to confirm**; the design works with any values.
7. **What users are told matches the reason.**
   - A voluntary move to Free sends `SUBSCRIPTION_DOWNGRADED`.
   - Payment retries running out sends `PAYMENT_EXHAUSTED`.
   - When the downgrade is scheduled and when it takes effect, if recent campaigns exceed the new limit, the message says existing campaigns stay live and when creating becomes possible again. The downgrade dialog shows the same information before the user confirms.

## Plan

1. **One limit function.** Rewrite `resolveEffectiveCampaignLimit` to use `currentPlan`, with 0 for `SUSPENDED_LEGAL` and `ACCOUNT_DEACTIVATED`.
   - Add `countCampaignsCreatedSince(userId, since)` and use it in `CampaignLimitService.enforceLimit` and in `getStatus`.
   - Keep a lock for concurrent creates: lock the `company_subscription` row instead of the billing period, so creating a campaign no longer fails when there is no active period.
   - Files: `SubscriptionService.java`, `CampaignLimitService.java`, `BillingPeriodRepository.java` (or the partnership repository), `CompanySubscriptionRepository.java` (add a `findByUserIdForUpdate`).
2. **Fix stale plan and state fields.**
   - `processTrialExpiryInTermsPending` must also set `currentPlan = FREE`. Today `acceptTerms` then leaves `FREE_ACTIVE` with an Enterprise plan, and the Free-period renewal copies Enterprise into the new period.
   - `handlePaymentFailed` saves `previousState` (only on the first failure). `handleInvoicePaid` restores it.
   - `requestDowngrade` rejects `PAYMENT_FAILED`.
   - Files: `SubscriptionService.java`.
3. **One way to finish a downgrade.**
   - Extract `completeDowngradeToFree(sub, reason)`, which ends the period, sets `FREE_ACTIVE`, clears target/schedule/subscription IDs, logs the event and picks the notification.
   - Call it from `handleSubscriptionDeleted`: reason VOLUNTARY if the state was `DOWNGRADE_PENDING`, PAYMENT_EXHAUSTED if it was `PAYMENT_FAILED`.
   - Replace the dead `PENDING_DOWNGRADE` query with the reconciliation query from rule 5.
   - Files: `SubscriptionService.java`, `BillingPeriodRepository.java`, `CompanySubscriptionRepository.java`, `SubscriptionPeriodProcessorCronJob.java`.
4. **Status DTO and plans endpoint.**
   - Add `campaignsUsedLast30Days` and `nextCreationAvailableAt` to the status DTO. Keep `campaignsUsedThisPeriod` as a deprecated alias for one release.
   - Add `GET /subscription/plans` (not gated on payments), returning name, price and limit.
   - Files: `SubscriptionStatusDtoOut.java`, `SubscriptionController.java`, `docs/openapi/openapi.json`.
5. **Notifications.**
   - Add `newLimit` and `recentCampaigns` parameters, and new translation rows for the "over the limit" variants of DOWNGRADE_SCHEDULED and DOWNGRADED.
   - Files: `SubscriptionService.java`, a new Liquibase changeset plus `changelog.xml`, `NotificationEmailService.java` (template map).
6. **Frontend plan and billing screens.**
   - Replace `PLAN_LADDER` limits/prices and `UPGRADE_PLANS` with the plans endpoint, keeping feature keys local.
   - Rename the usage label to "Campaigns created in the last 30 days".
   - Downgrade dialog: pass current usage and the target limit, and show the impact text plus the date creation becomes possible again.
   - Files: `plan-billing.component.ts/html`, `downgrade-confirm-dialog.component.ts/html`, `core/subscription/subscription.service.ts`, `core/api-frozen/hidden-models.ts` + `subscription.client.ts`, `assets/i18n/en.json` + `pl.json`.
7. **Frontend campaign creation.** Map error code `CAMPAIGN_LIMIT` to the existing, unused `subscription.campaign_limit` dialog with an upgrade link. Fix the `trial_offer` copy so it uses the live limit.
   - Files: the campaign-create feature component (not read), `en.json`/`pl.json`.
8. **Fixtures and contracts.** Update `demo-fixtures.ts`, `sandbox/fixtures/plan-billing.fixture.ts`, `testing/contract/subscription.contract.ts`, `e2e-tests/_framework/api/subscription.api.ts`, and the visual snapshots.

## Risks and invariants

| Invariant | Guarding test |
|---|---|
| **I1.** A plan change or payment event never changes `PartnershipOpportunity.active` | New: `SubscriptionService_Downgrade_IntegrationTest`, "downgrade to FREE with 8 recent campaigns leaves all active" |
| **I2.** The limit comes from `currentPlan` only; 0 for suspended/deactivated | `CampaignLimitServiceUnitTest` (update the plan-specific cases); new: `TERMS_PENDING` with stale `previousPlan=ENTERPRISE` on a FREE plan gives 5 |
| **I3.** The usage count survives a plan change (no reset) | New: `CampaignLimitServiceUnitTest`, "count unchanged across handleSubscriptionDeleted" |
| **I4.** A `FREE_ACTIVE` account always has `currentPlan=FREE` | New: `SubscriptionService_Terms_IntegrationTest`, "trial expires in TERMS_PENDING → accept → FREE plan"; add a check to `AdminIntegrityChecker` |
| **I5.** Payment recovery restores the earlier state, including `DOWNGRADE_PENDING` with its target | Extend `SubscriptionService_DowngradeUnitTest` "handleInvoicePaid — payment recovery" |
| **I6.** Each move to Free sends exactly one notification with the right type | Extend `SubscriptionService_WebhookUnitTest` (deleted while `DOWNGRADE_PENDING` vs `PAYMENT_FAILED`) |
| **I7.** The frontend never shows hardcoded limits | `plan-billing.component.spec.ts`: cards render limits from a mocked plans response; `subscription.contract.ts` |
| **I8.** Concurrent creates can't exceed the limit | Existing lock semantics; new integration test with two parallel creates at limit−1 |

**Risks**
- **Rolling window vs. copy.** The rolling window changes what "a month" means in the copy, and Polish/English terms text may promise per-billing-period behaviour. Check with legal before release.
- **Stale plan data in production.** Accounts may already be in the state from step 2. Ship a one-off migration that sets `current_plan_id = FREE` where `status = 'FREE_ACTIVE'`.
- **Old frontends.** Deprecating `campaignsUsedThisPeriod` needs a release where both fields are returned.
- **Payments disabled.** Most of this code is inactive while `app.payments.enabled=false` (the boot guard). A side issue I noticed: `enterTermsPending` moves Free accounts into `TERMS_PENDING`, which the boot guard counts as an in-flight paid status. Publishing new terms with payments off would stop the next startup. Handle it separately.

## Evidence

- **FACT:** the limit is checked only when creating a campaign. `PartnershipOpportunityService.java:805-808`; the only caller of `enforceLimit` in `src/main` (grep).
- **FACT:** the count is of campaigns created in the billing period, with no `active` filter. `BillingPeriodRepository.java:28-33`; `PartnershipOpportunity.java:143`.
- **FACT:** nothing deactivates campaigns on downgrade or failure. None of the handlers or crons in `SubscriptionService.java` touch `PartnershipOpportunity`.
- **FACT:** moves to Free start a new period at now. `SubscriptionService.java:369-371`, `422-424`, `574-575`, `612-613`.
- **FACT:** limits are FREE 2→5, BUSINESS 5, ENTERPRISE 10 in the database; the frontend ladder says FREE=2. `22-03-2026-subscription-tables.sql:192-195`; `09-04-2026-update-free-plan-campaign-limit.sql:8`; `plan-billing.component.ts:71-77`.
- **FACT:** trial banner copy promises "10 campaigns/month". `en.json:6585`.
- **FACT:** in-between states read `previousPlan` for the limit. `SubscriptionService.java:736-743`. Moves to Free set `previousPlan` and never clear it: 361, 569, 605.
- **INFERENCE:** a Free account with a stale `previousPlan` that enters `TERMS_PENDING` gets the old paid limit. Follows from 736-743 and 814-815.
- **FACT:** `processTrialExpiryInTermsPending` sets only `previousState` (630). `acceptTerms` restores only the status (840-845). The Free renewal uses `sub.getCurrentPlan()` (684).
- **INFERENCE:** the result is `FREE_ACTIVE` with the Enterprise plan and limit 10.
- **FACT:** `PENDING_DOWNGRADE` is never written in `src/main`, only read (`BillingPeriodRepository.java:25`). Grep shows it only in tests, docs and the constraint, so `processExpiredDowngrades` can never match.
- **FACT:** a voluntary downgrade to Free ends in `handleSubscriptionDeleted`, which logs and sends PAYMENT_EXHAUSTED ("All payment attempts … failed"). `SubscriptionService.java:483-487`, `372-376`; `23-03-2026-subscription-notification-translations.sql:37-38`.
- **FACT:** `handlePaymentFailed` doesn't record the earlier state (327-329). Recovery sets a status derived from the plan and leaves `targetPlan` alone (275-278). `cancelDowngrade` requires `DOWNGRADE_PENDING` (517-518).
- **INFERENCE:** a downgrade pending before the failure leaves the banner showing after recovery, and cancelling it fails with "no pending".
- **FACT:** that error would come from an `IllegalStateException` handler (`BusinessExceptionHandler.java:270`), and the frontend maps 409 to `no_pending` (`plan-billing.component.ts:550`).
- **HYPOTHESIS:** that handler returns 409. I didn't read its body.
- **FACT:** the backend allows a downgrade from `PAYMENT_FAILED` (472-475); the frontend offers none there (`plan-billing.component.ts:205-219`).
- **FACT:** `PAYMENT_FAILED` keeps `currentPlan` (327-329), as do `requestDowngrade` (498-501) and `enterTermsPending` (814-818). This is the basis for Proposed change rule 3.
- **FACT:** the frontend has no handling for `CAMPAIGN_LIMIT`; the `subscription.campaign_limit` keys are unused. Grep of `frontend/src` outside i18n; `en.json:6576-6583`; error code at `CampaignLimitExceededException.java:13`.
- **FACT:** the downgrade dialog shows only the target plan and date. `downgrade-confirm-dialog.component.html:17-33`.
- **FACT:** the paid-plan cron branches only run when payments are on; the boot guard counts `TERMS_PENDING`. `SubscriptionPeriodProcessorCronJob.java:36-41`; `PaymentsDisabledBootGuard.java:35-41`.
- **FACT:** `enterTermsPending` includes `FREE_ACTIVE`. `SubscriptionService.java:801-804`.
- **HYPOTHESIS:** the campaign-create component shows a generic error for `CAMPAIGN_LIMIT`. I didn't read it.
- **HYPOTHESIS:** Stripe retries keep going for a configured dunning window before sending `subscription.deleted`. This is Stripe dashboard configuration, not in the repo.
