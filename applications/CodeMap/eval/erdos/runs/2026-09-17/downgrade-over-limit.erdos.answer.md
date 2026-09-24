## Problem
A company on Enterprise (or on the Enterprise trial) that has more campaigns than Free allows can move to Free in three ways: it downgrades on purpose, its trial ends, or its payments run out. I traced what the backend and frontend do in each case and where the rules disagree, and I propose one policy with a plan to implement it. It's done when every path to Free uses one limit rule and one way of counting, sends the right event and notification, respects the terms-consent state, and the frontend tells the user what will happen before and after.

## Where it lives today
**Subsystems:** backend [11] Subscriptions, [4] Partnership opportunities, [12] Notifications; frontend [174] Plan & billing, [171] Opportunities.

**Plan limits** (backend `src/main/resources/db/changelog/2026/03/22-03-2026-subscription-tables.sql:192` and `2026/04/09-04-2026-update-free-plan-campaign-limit.sql:8`): Free 5 (raised from 2), Business 5, Enterprise 10. Only Enterprise or a trial can be over the Free limit, and Business → Free changes no limit.

**The limit is checked only when a campaign is created.**
- `PartnershipOpportunityService.java:806-807` calls `CampaignLimitService.enforceLimit`.
- That method counts `PartnershipOpportunity.createdTime BETWEEN period.start AND period.end` (`BillingPeriodRepository.countCampaignsInPeriod`).
- So it counts campaigns **created in the billing period**, not **active** campaigns. A campaign has no status field, only `endDate`, so "active" isn't modelled anywhere.
- Nothing on any downgrade path touches existing campaigns.

**How the state changes** (`subscription/SubscriptionService.java`):

| Path | What happens today |
|---|---|
| Downgrade request, `requestDowngrade` (442-509) | Allowed from BUSINESS_ACTIVE, ENTERPRISE_ACTIVE or PAYMENT_FAILED. Sets `DOWNGRADE_PENDING`, `previousPlan = currentPlan`, `targetPlan`. `currentPlan` stays unchanged. For Free it tells Stripe to cancel at period end. |
| Cron `processExpiredDowngrades` (591-622), run by `SubscriptionPeriodProcessorCronJob` at 04:00 when payments are on | Selects billing periods with status `PENDING_DOWNGRADE`. **No main code ever sets that status** (only tests do), so this cron never does anything in production. It also sends no notification. |
| Stripe `customer.subscription.deleted` → `handleSubscriptionDeleted` (340-377) | Today this is the only real way to Free, for **both** a voluntary downgrade and payment exhaustion. It always records `PAYMENT_EXHAUSTED` and sends `SUBSCRIPTION_PAYMENT_EXHAUSTED`. It sets `FREE_ACTIVE` unconditionally, even if the company is in `TERMS_PENDING`. |
| `invoice.payment_failed` → `handlePaymentFailed` (316-337) | `PAYMENT_FAILED`, `previousPlan = currentPlan`; the paid plan stays. No deadline of our own: the exit is Stripe's retries ending in `invoice.paid` (recovery) or `subscription.deleted`. |
| `cancelDowngrade` (512-550) | Always restores an *active* status, even when the downgrade was requested from `PAYMENT_FAILED`. The payment failure is hidden. |
| Enterprise→Business schedule → `handleSubscriptionUpdated` (380-435) | New billing period from now, so the counter resets. |
| Trial expiry or trial cancel (557-588, 447-470) | Immediately Free, new period, notification. |
| `invoice.paid` → `handleInvoicePaid` (287-290) | **Pushes `endDate` of the active period back one month; `startDate` never moves.** On paid plans the usage count therefore adds up across renewals, while Free periods roll over monthly (`renewExpiredFreeBillingPeriods`). |
| `enterTermsPending` (800-829) | Saves `previousState`, leaves `currentPlan` and `previousPlan` as they were. |

**Which limit applies**, `resolveEffectiveCampaignLimit` (734-748): in DOWNGRADE_PENDING, PAYMENT_FAILED and TERMS_PENDING it uses `previousPlan`, in SUSPENDED_LEGAL it returns 0, otherwise `currentPlan`.
- In the first two states `previousPlan == currentPlan`, so the branch does nothing.
- In TERMS_PENDING `previousPlan` is left over from the last change. An Enterprise payer who upgraded from Free gets **Free's limit**; a Free company that came down from Enterprise gets **10**.

**Frontend**
- `feature/plan-billing/plan-billing.component.html:107-194` shows `campaignsUsedThisPeriod / campaignLimit`, a payment-failed banner and a pending-downgrade banner.
- The downgrade dialog text (`en.json:379`) says "You'll keep your current benefits until then". It says nothing about campaigns over the Free limit.
- `feature/opportunities/opportunity-form.component.ts:564-570` handles only 404. A 409 `CAMPAIGN_LIMIT` becomes the generic `save_failed` message.
- The showcase caption `en.json:2908` says "the period-end cron applies it", which the code contradicts.

## Proposed change
**One policy.**
1. **The limit caps how many campaigns can be created per monthly billing period, the same way on every plan.** Periods are always one month long and a renewal starts a new period; it doesn't extend the old one.
2. **The limit follows `currentPlan`.** It changes only when the plan change takes effect (end of a downgraded period, or payments exhausted). `SUSPENDED_LEGAL` and `ACCOUNT_DEACTIVATED` get 0. `TERMS_PENDING` uses `currentPlan`.
3. **Existing campaigns are kept as they are when a company moves to Free.** They are never closed or hidden, and applied-opportunity flows (including `TO_BE_PAID`) continue. From then on the company can create up to 5 campaigns per period.
4. **One method applies the move to Free**, idempotently. It records the reason: voluntary (`SUBSCRIPTION_DOWNGRADED`) or payments exhausted (`PAYMENT_EXHAUSTED`). The Stripe webhook calls it; the cron calls it as a fallback when the webhook is missed. If the company is in `TERMS_PENDING`, it changes the plan and `previousState` and keeps `TERMS_PENDING`.
5. **Cancelling a downgrade restores the state it was requested from.**

**Alternative considered:** a cap on concurrently active campaigns, with automatic archiving on downgrade. I rejected it. `PartnershipOpportunity` has no status to archive into. Archiving would strand influencers in accepted, content or `TO_BE_PAID` states, which are payment obligations. It would also mean a new campaign state machine. If product wants a concurrent cap later, the only place to change is `CampaignLimitService`.

## Plan
1. **Which limit applies** (safe on its own). `resolveEffectiveCampaignLimit` returns `currentPlan.campaignLimit`, or 0 for SUSPENDED_LEGAL and ACCOUNT_DEACTIVATED.
   - Files: `SubscriptionService.java`, `CampaignLimitServiceUnitTest`, a new case in `SubscriptionService_*` tests for TERMS_PENDING after an upgrade.
2. **Restore the right state on cancel.** `requestDowngrade` stores `previousState`, and `cancelDowngrade` restores it (PAYMENT_FAILED stays PAYMENT_FAILED).
   - Files: `SubscriptionService.java`, its downgrade unit tests.
3. **One way to Free.** Extract `applyFreeTransition(sub, reason, stripeEventId)`.
   - `handleSubscriptionDeleted` works out the reason from the status: DOWNGRADE_PENDING → voluntary, event `SUBSCRIPTION_DOWNGRADED`, notification `SUBSCRIPTION_DOWNGRADED`; otherwise `PAYMENT_EXHAUSTED`.
   - If the status is TERMS_PENDING, set `currentPlan = FREE` and `previousState = FREE_ACTIVE` and keep TERMS_PENDING, the same way `processTrialExpiryInTermsPending` does.
   - Rewrite `processExpiredDowngrades` to select subscriptions in `DOWNGRADE_PENDING` whose active period ended more than 24h ago, so the webhook wins. For Stripe-linked subscriptions, first confirm with `stripeService.retrieveSubscription` that Stripe has cancelled it, then call the same method.
   - Delete the unused `findExpiredPendingDowngrades` and the `PENDING_DOWNGRADE` enum value, or keep the value because of the DB check constraint.
   - Files: `SubscriptionService.java`, `BillingPeriodRepository.java`, `CompanySubscriptionRepository.java`, `SubscriptionService_CronUnitTest`, `SubscriptionService_Cron_IntegrationTest`, `StripeWebhookControllerUnitTest`.
4. **Billing periods.** `handleInvoicePaid` ends the active period and creates `[oldEnd, oldEnd+1M)` instead of pushing `endDate` back. It must skip the first invoice of a subscription, which `handleCheckoutCompleted` already covers (use the invoice's `billing_reason` of `subscription_create`, passed through from `StripeWebhookHandler`).
   - Data migration (Liquibase SQL under `db/changelog/2026/09/`): for ACTIVE paid periods longer than one month, set `start_date = end_date - 1 month`.
   - Files: `SubscriptionService.java`, `stripe/StripeWebhookHandler.java`, a new changelog file, the invoice-paid tests.
5. **More status data for the frontend.** Add `activeCampaigns` (`endDate IS NULL OR endDate >= now`, a new repository query) and `targetPlanCampaignLimit` to `SubscriptionStatusDtoOut`.
   - Files: `SubscriptionStatusDtoOut.java`, `BillingPeriodRepository.java` or the opportunity repository, `SubscriptionService.getStatus`; OpenAPI contract tests in [16].
6. **Frontend.**
   - Add the new fields to `core/api-frozen/hidden-models.ts` (and regenerate `api` if the endpoint is generated).
   - `downgrade-confirm-dialog.component.{ts,html}` warns when `activeCampaigns > targetPlanCampaignLimit`: existing campaigns stay live, and the company can create N per month on Free.
   - `opportunity-form.component.ts` maps a 409 whose body `error === 'CAMPAIGN_LIMIT'` to a new key with a link to plan & billing.
   - `plan-billing.component.html` shows a notice after payments are exhausted, based on `SUBSCRIPTION_PAYMENT_EXHAUSTED` or the status.
   - `assets/i18n/en.json` and `pl.json`: new keys, and fix the captions at 2896/2908.
   - Update `sandbox/fixtures/plan-billing.fixture.ts` and `downgrade-confirm.fixture.ts`, plus specs.
   - Deploy after the backend step 5.
7. **E2E.** Extend `frontend/e2e-tests/integration/flows/subscription-lifecycle.spec.ts`: Enterprise with 7 campaigns → downgrade → `subscription.deleted` → campaigns stay visible, the 6th creation in the new period is blocked, and the notification is the downgrade one, not payment-exhausted.

## Risks and invariants
- **Consent:** a Stripe event must never move a company out of `TERMS_PENDING` or `SUSPENDED_LEGAL`. Test: `subscription.deleted` while in TERMS_PENDING leaves the status TERMS_PENDING with `previousState=FREE_ACTIVE`.
- **Idempotency:** a repeated `subscription.deleted`, or the cron running after the webhook, must not create a second Free period or send a second notification. Guard: the status check inside the method, plus the row lock from `findBy…ForUpdate`. Test: call it twice and expect one period.
- **Race between cron and webhook:** the 24h delay plus the Stripe check. Test: the cron skips when Stripe still reports the subscription as active.
- **Existing campaigns are never changed by a subscription change.** Test: opportunity rows and applied-opportunity statuses are unchanged after each move to Free.
- **Payments off (`app.payments.enabled=false`):** the Free rollover and step 1 must keep working. Guard: `SubscriptionService_PaymentsDisabled_IntegrationTest`.
- **Migration in step 4:** it shortens current paid periods, so a company could suddenly find itself "over quota" in the current period. That's acceptable: it only blocks new creation, nothing is removed. Announce it.
- **Frontend contract:** the frozen models must match the DTO. Guard: the OpenAPI contract tests in [16].

## Evidence
- FACT — the limit counts campaigns *created* in the period, and is checked only at creation: `CampaignLimitService.java:21-37`, `BillingPeriodRepository.java:28-30`, `PartnershipOpportunityService.java:805-808`.
- FACT — plan limits Free 5, Business 5, Enterprise 10: the two changelog files above.
- FACT — `PENDING_DOWNGRADE` is written only in tests: a grep over `backend/src` found main-code hits only in the enum, the repository query and the DDL.
- FACT — `handleSubscriptionDeleted` always records PAYMENT_EXHAUSTED and forces FREE_ACTIVE: lines 340-377.
- FACT — `cancelDowngrade` always restores an active status: lines 533-538, with 472-474 allowing a downgrade from PAYMENT_FAILED.
- FACT — `handleInvoicePaid` pushes `endDate` back without moving `startDate`: lines 287-290.
- FACT — `enterTermsPending` doesn't set `previousPlan`, and the effective limit reads `previousPlan` in TERMS_PENDING: lines 800-829 and 734-748.
- FACT — `PartnershipOpportunity` has `endDate` but no status field (grep of the entity); `OpportunityStatus` belongs to AppliedOpportunity.
- FACT — the frontend downgrade text promises unchanged benefits, and the opportunity form maps only 404: `en.json:379`, `opportunity-form.component.ts:564-570`.
- INFERENCE — the 409 CAMPAIGN_LIMIT is shown to the user only as a generic save error, unless a global error interceptor shows the backend message. To verify.
- INFERENCE — Stripe sends `customer.subscription.deleted` at period end after `cancel_at_period_end`. That is Stripe's documented behaviour; our code relies on it at `requestDowngrade:485-487`.
- HYPOTHESIS — the webhook layer already removes duplicate Stripe event ids. To verify.
- HYPOTHESIS — the first `invoice.paid` can arrive after checkout is linked and add an extra month to the first period. It depends on webhook order; step 4 handles either order.

## Corrections after verification

## Corrections and confirmations

1. **INFERENCE → FACT: a 409 `CAMPAIGN_LIMIT` reaches the user only as "save failed".** The only global error handler, `frontend/src/app/core/interceptors/error.interceptor.ts:84-86`, passes through every status except the ones that trigger a session refresh. So a 409 goes straight to `opportunity-form.component.ts`, which shows the generic `opportunities.form.error.save_failed`. Plan step 6 stands.

2. **HYPOTHESIS → FACT, with a caveat for step 3: Stripe webhooks are already deduplicated.**
   - `StripeWebhookHandler.java:42-52` checks `existsByStripeEventId` first. The real guard is the unique constraint `subscription_event_stripe_unique`, which fires when `logEvent` writes the event id.
   - **Caveat:** the new `applyFreeTransition` must still call `logEvent` with the Stripe event id. Otherwise webhook dedup is lost.
   - The fallback cron run has no Stripe event id, so the constraint doesn't protect it. Only the "status is still DOWNGRADE_PENDING" check under the row lock does. Load it with a `…ForUpdate` query, not the plain `findByUserId` that `processExpiredDowngrades` uses today.

3. **Still INFERENCE, unchanged: Stripe sends `customer.subscription.deleted` at period end.** This is Stripe's documented behaviour and can't be settled from this code. The cron's 24h fallback in step 3 exists for exactly this case.

4. **Still HYPOTHESIS, unchanged: the first `invoice.paid` may add a month to the first period.** It depends on webhook order at runtime, which the code can't settle. Step 4 skips invoices with `billing_reason=subscription_create`, so the plan works either way.
