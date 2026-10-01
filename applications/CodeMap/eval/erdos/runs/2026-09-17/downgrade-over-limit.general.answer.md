## Problem

Today "downgrading" only changes which limit applies to **creating** a campaign. Nothing ever looks at the campaigns a company already has. The backend and frontend also disagree on the limits themselves and on what should happen when the user cancels versus when their payment fails. Because the rules are spread across webhook handlers, cron jobs, the screens and the email text, each exit path (planned downgrade, payment failures running out, trial end, trial cancel, terms suspension) does a slightly different thing.

## Where it lives today

**Backend (`backend/src/main/java/com/sm/instagram/platform/…`)**
- `subscription/SubscriptionService.java`: the state machine.
  - `requestDowngrade` (441–509), `cancelDowngrade` (511–550)
  - `handlePaymentFailed` (315–337), `handleInvoicePaid` (259–313), `handleSubscriptionDeleted` (339–377), `handleSubscriptionUpdated` (379–435)
  - `processExpiredTrials` (556–588), `processExpiredDowngrades` (590–622), `renewExpiredFreeBillingPeriods` (669–702)
  - `resolveEffectiveCampaignLimit` (734–748)
- `subscription/CampaignLimitService.java`: the only place a limit is enforced (`count >= limit` → error). It is called from `partnershipopportunities/PartnershipOpportunityService.java:806-808`, on create only.
- `subscription/repository/BillingPeriodRepository.java`: `countCampaignsInPeriod` counts every campaign created in the period, including soft-deleted ones. `findExpiredPendingDowngrades` looks for billing periods with status `PENDING_DOWNGRADE`.
- `subscription/stripe/StripeWebhookHandler.java`: routes `invoice.paid`, `invoice.payment_failed`, `customer.subscription.deleted` and `customer.subscription.updated` (the last only when `items` or `schedule` changed).
- `subscription/stripe/StripeService.java`: a downgrade to FREE sets `cancel_at_period_end`; Enterprise→Business uses a subscription schedule with `RELEASE`.
- `subscription/cron/SubscriptionPeriodProcessorCronJob.java`: daily at 04:00 runs trials, then downgrades, then FREE period renewal.
- `notification/NotificationType.java:240-249`, `notification/email/NotificationEmailService.java:85-94`, `resources/messages_{en,pl}.properties:657-675`.
- `resources/db/changelog/2026/03/22-03-2026-subscription-tables.sql:192-195` seeds the limits FREE 2 / BUSINESS 5 / ENTERPRISE 10; `2026/04/09-04-2026-update-free-plan-campaign-limit.sql` raises FREE to **5**.

**Frontend (`frontend/src/…`)**
- `app/feature/plan-billing/plan-billing.component.ts`: `PLAN_LADDER` hardcodes limits **2/5/10** and prices; `availableDowngrades` only offers downgrades from `*_ACTIVE`.
- `app/feature/plan-billing/plan-billing.component.html:106-212`: usage ring (used / limit), payment-failed alert, pending-downgrade banner.
- `app/feature/plan-billing/downgrade-confirm-dialog.component.{ts,html}`: says nothing about campaigns.
- `assets/i18n/{en,pl}.json` (`plan_billing.*`, `campaign_limit.*`), `app/core/demo/demo-fixtures.ts:1014-1022`, `app/sandbox/fixtures/plan-billing.fixture.ts`.

**What each flow does today**
1. **Planned downgrade to FREE:** the status becomes `DOWNGRADE_PENDING` and Stripe is set to cancel at period end. At period end Stripe sends `customer.subscription.deleted`, which `handleSubscriptionDeleted` treats as payment failure: it logs `PAYMENT_EXHAUSTED` and sends the "All payment attempts … have failed" email. The cron job that was meant to apply this downgrade never matches anything, because nothing in main code ever sets a billing period to `PENDING_DOWNGRADE`.
2. **Payment fails:** the status becomes `PAYMENT_FAILED` and the paid limit stays in place until Stripe gives up and deletes the subscription. Then the account falls back to FREE with a new FREE period starting now. The count starts again from 0 and existing campaigns stay live.
3. **Downgrade Enterprise→Business:** the schedule changes the subscription items, and `handleSubscriptionUpdated` switches the plan and starts a new period.
4. **Over the limit after any fallback:** existing campaigns are never touched, and creating new ones is blocked only after 5 more are created in the new period. No screen or email tells the user.

**Inconsistencies found**

| # | Inconsistency | Where |
|---|---|---|
| A | FREE limit is 2 on the screen and in the trial-expired email, 5 in the database | `plan-billing.component.ts:77`, `messages_en.properties:657` vs the 09-04 migration |
| B | A planned downgrade to FREE is reported as "payment attempts failed", and the "Plan downgraded" notification is never sent | `handleSubscriptionDeleted` |
| C | `processExpiredDowngrades` and `PENDING_DOWNGRADE` are never triggered in production; only tests exercise them | `BillingPeriodRepository:24-26` |
| D | In `TERMS_PENDING` the limit comes from `previousPlan`, which is the plan *before the last change* (checkout sets it to FREE). A paying Enterprise customer asked to accept new terms drops to the FREE limit. | `resolveEffectiveCampaignLimit:736-742`, `handleCheckoutCompleted:238` |
| E | `handlePaymentFailed` and `handleSubscriptionDeleted` overwrite `TERMS_PENDING` (and `DOWNGRADE_PENDING`), so the account slips out of the terms grace/suspension process: `findExpiredGracePeriods` filters on `status='TERMS_PENDING'`. The trial path handles this correctly (`processTrialExpiryInTermsPending`). | `SubscriptionService.java:327-328, 361-363` |
| F | Payment recovery restores `currentPlan = previousPlan`. If the plan changed during `PAYMENT_FAILED`, the account reverts to the old plan while Stripe keeps billing the new price. Recovery also ignores a pending downgrade: after `DOWNGRADE_PENDING → PAYMENT_FAILED → paid` the status becomes `*_ACTIVE` but `targetPlan` and the Stripe cancel/schedule remain. | `handleInvoicePaid:275-278`, `handleSubscriptionUpdated:415-417` |
| G | The backend allows downgrade from `PAYMENT_FAILED`; the frontend hides it | `SubscriptionService:472-476` vs `plan-billing.component.ts:205-219` |
| H | `ACCOUNT_DEACTIVATED` still gets the full plan limit; only `SUSPENDED_LEGAL` gets 0 | `resolveEffectiveCampaignLimit` |
| I | Campaigns over the new limit are handled nowhere: no rule, no downgrade warning, no email explanation | whole stack |
| J | Prices and limits are hardcoded in the frontend with "keep in sync" comments, because the backend has no endpoint listing plans | `plan-billing.component.ts:71,163-169` |

## Proposed change: one policy, "keep existing campaigns, limit new ones, one path to FREE"

1. **What is counted.** The limit stays "campaigns created in the current billing period, including deleted ones", which is what the product copy says ("{{n}} campaigns a month"). The UI and emails state this explicitly.
2. **Existing campaigns are kept.** Changing plan never deactivates, hides or edits an existing campaign. Being over the limit only blocks creating new ones. This is the only safe choice: `deactivateOpportunity` refuses campaigns that have active applications, and removing live campaigns would break creator applications.
3. **One rule for the limit.** The limit is always `currentPlan.campaignLimit`, except `SUSPENDED_LEGAL` and `ACCOUNT_DEACTIVATED`, which get 0. This works because `currentPlan` never changes while in `DOWNGRADE_PENDING` or `PAYMENT_FAILED`: those states keep the paid plan until the change takes effect. `previousPlan` stops affecting the limit, which fixes D and H.
4. **One path to FREE.** Add `applyFreeFallback(sub, FallbackReason, stripeEventId)` with reasons `SCHEDULED_DOWNGRADE`, `PAYMENT_EXHAUSTED`, `TRIAL_EXPIRED`, `TRIAL_CANCELLED`. It is the only code that moves an account to FREE. It closes the old period, sets the plan, clears Stripe IDs and `targetPlan`, opens a new FREE period, and logs the event and notification that match the reason. The notification includes `newLimit` and `liveCampaigns`.
   - If the account is in `TERMS_PENDING`, it records FREE in `previousState` instead of overwriting the status. The same applies to `PAYMENT_FAILED`. This fixes E.
   - For `customer.subscription.deleted`, the reason is `SCHEDULED_DOWNGRADE` when the account is `DOWNGRADE_PENDING` with target FREE, or when Stripe's cancellation reason is "cancellation requested" (covers cancels made in the Stripe portal). Otherwise it is `PAYMENT_EXHAUSTED`. This fixes B.
5. **Payment-failure states don't override each other.**
   - `handlePaymentFailed` no longer changes `previousPlan`.
   - Recovery sets the status from `currentPlan`, or back to `DOWNGRADE_PENDING` if `targetPlan != null`.
   - `handleSubscriptionUpdated` while in `PAYMENT_FAILED` updates the plan but keeps `PAYMENT_FAILED`.
   - `requestDowngrade` from `PAYMENT_FAILED` is rejected with 409, matching the frontend. The fix is updating the payment method in the portal, or waiting for the retries to end.
   
   This fixes F and G.
6. **Replace the dead cron job with a safety net.** A reconciliation job looks for `DOWNGRADE_PENDING`/FREE accounts whose period ended more than 48 h ago, checks the subscription in Stripe, and if it is canceled calls `applyFreeFallback(SCHEDULED_DOWNGRADE)`. It can safely run more than once. `PENDING_DOWNGRADE` stays in the enum and constraint, just unused. This fixes C.
7. **The backend is the only source of limits.**
   - A new always-on `GET /subscription/plans` returns `[name, price, campaignLimit]`.
   - `SubscriptionStatusDtoOut` gains `liveCampaigns` (`active=true` and not ended).
   - The frontend drops its hardcoded values, and the downgrade dialog shows the effect: "You have N live campaigns — they stay live. From {date} you can create {limit} per period." This fixes A, I and J.

**Why this approach:** it keeps the enforcement contract and the enum/schema unchanged, avoids touching campaign data, and turns five different fallback implementations into one tested method.

## Plan

1. **Limit rule:** rewrite `resolveEffectiveCampaignLimit` as described. Touches `SubscriptionService.java` and a new `unit/service/subscription/SubscriptionService_CampaignLimitRuleUnitTest.java`.
2. **`FallbackReason` + `applyFreeFallback`:** route `handleSubscriptionDeleted`, `processExpiredTrials` and the trial-cancel branch of `requestDowngrade` through it, with notifications per reason. Touches `SubscriptionService.java`, new `subscription/entity/FallbackReason.java`.
3. **Pass Stripe's cancellation reason to `handleSubscriptionDeleted`.** Touches `StripeWebhookHandler.java`, `SubscriptionService.java`.
4. **Payment-failure fixes** (policy 5). Touches `SubscriptionService.java` (`handlePaymentFailed`, `handleInvoicePaid`, `handleSubscriptionUpdated`, `requestDowngrade`) and `SubscriptionService_WebhookUnitTest.java`, `SubscriptionService_DowngradeUnitTest.java`.
5. **Replace `processExpiredDowngrades` with `reconcileStaleFreeDowngrades`:**
   - Touches `SubscriptionService.java`, `SubscriptionPeriodProcessorCronJob.java`, `BillingPeriodRepository.java` (remove `findExpiredPendingDowngrades`) and `CompanySubscriptionRepository.java` (add the query).
   - Rewrite the matching cases in `SubscriptionService_CronUnitTest.java`, `SubscriptionService_Cron_IntegrationTest.java` and `SubscriptionRepository_Query_IntegrationTest.java`.
6. **Plans endpoint and `liveCampaigns`.** Touches `SubscriptionController.java`, new `dto/PlanDtoOut.java`, `SubscriptionStatusDtoOut.java`, `PartnershipOpportunityRepository.java` (a `countLive` query), `getStatus`.
7. **Email text:** fill in the limit as a parameter instead of "2", and add "your existing campaigns stay live" to `trial_expired`, `downgraded` and `payment_exhausted`; add a separate scheduled-downgrade message. Touches `messages_{en,pl}.properties`, `NotificationEmailService.java` (parameter mapping).
8. **Frontend models and client:** add `getPlans` and `liveCampaigns`. Touches `core/api-frozen/hidden-models.ts`, `core/api-frozen/subscription.client.ts`, `core/subscription/subscription.service.ts`, `testing/contract/subscription.contract.ts`.
9. **Plan and billing screen:** build `PLAN_LADDER` and `UPGRADE_PLANS` from `getPlans()`. Touches `plan-billing.component.{ts,html,spec.ts}`.
10. **Downgrade dialog impact text.** Touches `downgrade-confirm-dialog.component.{ts,html,spec.ts}`, `assets/i18n/{en,pl}.json` (`plan_billing.downgrade.confirm.*`, `plans.downgrade_note`).
11. **Fixtures and end-to-end tests.** Touches `core/demo/demo-fixtures.ts` (+spec), `sandbox/fixtures/{plan-billing,downgrade-confirm}.fixture.ts`, `e2e-tests/integration/flows/subscription-lifecycle.spec.ts`, `e2e-tests/bdd/steps/subscription.steps.ts`.

## Risks and invariants

- **I1. The limit is a pure function of status and `currentPlan`, and suspended or deactivated accounts get 0.** Guarded by a table-driven test over all 9 statuses, with `previousPlan` deliberately set to a different plan.
- **I2. No plan change deactivates a campaign.** Guarded by an integration test: Enterprise with 8 live campaigns → each fallback reason → all 8 still `active=true`, and `enforceLimit` throws at the 5th new campaign.
- **I3. Exactly one path to FREE, with the right notification.** Guarded by webhook unit tests: `deleted` while `DOWNGRADE_PENDING` sends `SUBSCRIPTION_DOWNGRADED` and never `PAYMENT_EXHAUSTED`; `deleted` from `PAYMENT_FAILED` sends `PAYMENT_EXHAUSTED`.
- **I4. `TERMS_PENDING` is never overwritten by a webhook.** Guarded by unit tests for `paymentFailed` and `deleted` while in `TERMS_PENDING`, checking `previousState` and that `findExpiredGracePeriods` still picks the account up.
- **I5. Recovery never changes `currentPlan`, and a pending downgrade survives a payment failure.** Guarded by `PAYMENT_FAILED → subscription.updated → invoice.paid` and `DOWNGRADE_PENDING → failed → paid` tests.
- **I6. The frontend renders only backend-supplied limits and prices.** Guarded by a plan-billing spec that stubs `getPlans` with odd values (3/7/11) and checks they appear on the cards.

**Risks**
- **Usage count resets on immediate fallbacks.** Trial cancel and payment-exhausted fallbacks start a fresh FREE period, so the count resets. This is accepted and documented, since it matches normal period renewal.
- **Stripe event order.** Stripe doesn't guarantee delivery order, so the reconciliation job must not race the webhook. Both paths lock the subscription row (`findBy…ForUpdate` / `@Version`), and `applyFreeFallback` does nothing if the account is already `FREE_ACTIVE`.
- **Deploy order.** Old frontends still show 2. Deploy the backend first; the frontend change only adds to what it reads.

## Evidence

**Read in the code (FACT)**
1. Enforcement happens only when creating a campaign, as `count >= limit` over campaigns created in the period (`CampaignLimitService.java:21-37`, `PartnershipOpportunityService.java:806`). The count query has no `active` filter (`BillingPeriodRepository.java:28-33`).
2. Seeded limits are FREE 2 / BUSINESS 5 / ENTERPRISE 10, and the 09-04 migration sets FREE to 5. The frontend shows 2 (`plan-billing.component.ts:77`) and the email says "Free (2 campaigns/month)" (`messages_en.properties:657`).
3. `handleSubscriptionDeleted` always logs `PAYMENT_EXHAUSTED` and sends that notification (`SubscriptionService.java:372-376`). A downgrade to FREE sets `cancel_at_period_end` (`:486`, `StripeService.java:140-146`).
4. `PENDING_DOWNGRADE` is set only in tests; main code only reads it (search across `backend/src`).
5. `resolveEffectiveCampaignLimit` uses `previousPlan` for `TERMS_PENDING`, `PAYMENT_FAILED` and `DOWNGRADE_PENDING`, and checkout sets `previousPlan` to the old plan (`:238`, `:736-747`).
6. `handlePaymentFailed` and `handleSubscriptionDeleted` set the status without checking for `TERMS_PENDING` (`:327-328`, `:361-363`). The grace-period query filters on `status='TERMS_PENDING'` (`CompanySubscriptionRepository.java:27-29`).
7. Recovery sets `currentPlan = previousPlan` (`:275-278`), and `handleSubscriptionUpdated` sets `previousPlan` and an active status (`:415-417`).
8. The backend allows downgrade from `PAYMENT_FAILED` (`:472-476`); the frontend doesn't offer it (`plan-billing.component.ts:205-219`).
9. `deactivateOpportunity` refuses campaigns with active applications (`PartnershipOpportunityService.java:1479-1487`).
10. `ACCOUNT_DEACTIVATED` isn't given 0 by the limit function (`:734-748`).
11. The webhook handler ignores `subscription.updated` unless `schedule` or `items` changed (`StripeWebhookHandler.java:154-171`).

**Inferred or unconfirmed**
12. INFERENCE (Stripe behaviour): a subscription set to `cancel_at_period_end` produces `customer.subscription.deleted` at period end.
13. HYPOTHESIS: Stripe `Subscription` objects carry a cancellation reason that the SDK in use exposes.
14. INFERENCE: nothing in the frontend handles the `CAMPAIGN_LIMIT` error; the `campaign_limit.*` translation keys are unused.
15. INFERENCE: `NotificationEmailService` fills in positional parameters (`{0}`, `{1}`) from the params map, so a limit parameter can be added.

## Corrections after verification

**Corrections:**

1. **Evidence 15 was wrong, so plan step 7 changes.** The `email.subscription.*.message` entries in `messages_{en,pl}.properties` are never read. `NotificationEmailService.java:106-111` reads only `.subject` and `.title`; the email body is `notification.getMessage()` (line 114). That text is translated by `NotificationService.java:98-104`, using named placeholders (`{oldPlan}`, `{newPlan}`) from the DB translations in `db/changelog/2026/03/23-03-2026-subscription-notification-translations.sql`.
   - So the "Free (2 campaigns/month)" line (`messages_en.properties:657`) is dead text that no user sees. It is a cleanup item, not live inconsistency A. Inconsistency A still holds for the screen (`plan-billing.component.ts:77`).
   - **Revised step 7:** add a new Liquibase changeset that updates or adds the `NOTIFICATION_SUBSCRIPTION_{TRIAL_EXPIRED,PAYMENT_EXHAUSTED,DOWNGRADED}_MESSAGE` rows (en and pl) with `{newLimit}` / `{liveCampaigns}` and the "existing campaigns stay live" sentence. Add a `SUBSCRIPTION_DOWNGRADED` wording for the scheduled-downgrade reason, keyed on the params map.
   - `NotificationEmailService.java` needs no change. Delete the unused `.message` keys from `messages_{en,pl}.properties`.
   - Parameter names must match the named placeholders, not `{0}`/`{1}`.

2. **Evidence 14 is confirmed (now FACT).** The `CAMPAIGN_LIMIT` error appears only in end-to-end tests (`e2e-tests/_framework/api/subscription.api.ts:121`, `bdd/steps/subscription.steps.ts:219-224`, `integration/flows/subscription-lifecycle.spec.ts:497-562`). The `campaign_limit.*` keys in `assets/i18n/{en,pl}.json:6576` are used nowhere under `src/app`.
   - **Addition to step 10:** have the campaign-create error path show those keys (used / limit / plan, with a link to plan and billing). Otherwise a downgraded company gets a 409 with no explanation.

3. **Evidence 13 is still a HYPOTHESIS.** The Stripe SDK is outside the workspace, so I couldn't check it. The design no longer depends on it:
   - Inside the app, the account's own state (`DOWNGRADE_PENDING` with target FREE) is enough to tell a planned downgrade from a payment failure.
   - For cancels made in the Stripe portal, extend `StripeWebhookHandler.handleSubscriptionUpdated` to also react when `cancel_at_period_end` changes, and mark the account `DOWNGRADE_PENDING`/FREE (or clear it). Then `subscription.deleted` is labelled correctly without Stripe's cancellation reason. This adds `StripeWebhookHandler.java` to step 3.

4. **Evidence 12 is still an INFERENCE** about how Stripe behaves. Nothing in the repo tests the "canceled at period end → `deleted`" flow end to end. The step 5 reconciliation job remains the backstop if that event is missing or never sent.
