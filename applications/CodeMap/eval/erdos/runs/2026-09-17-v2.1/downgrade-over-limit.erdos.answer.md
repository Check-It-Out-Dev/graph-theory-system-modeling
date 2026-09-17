I found how campaign limits are enforced across backend and frontend, and where the rules conflict. The main gap: **nothing** happens to a company's existing campaigns when it drops to a lower plan. The limit is not a cap on active campaigns. It counts campaigns created in the current billing period, and every drop to a lower plan opens a new period, so the count restarts at zero.

## Problem
A paid company (Enterprise allows 10 campaigns, Business 5) drops to Free (5). It gets there by choosing to downgrade, by its payments failing until Stripe deletes the subscription, or when its trial ends or is cancelled. The goal is one policy for what the company may still do, what its existing campaigns do, what it is told and what the plan and billing screen shows, applied the same way on every path.

It is done when:
- one backend function computes the limit that applies, and one computes usage;
- every state change applies them the same way;
- the plan and billing screen and the campaign form show the same numbers the backend enforces;
- tests pin each path.

## Where it lives today
**Backend, `backend/src/main/java/com/sm/instagram/platform/subscription/`:**
- **Limit check:** `CampaignLimitService.java` is called only from `partnershipopportunities/PartnershipOpportunityService.java:806-808`, when a campaign is created.
- **Counting:** `repository/BillingPeriodRepository.java`. `countCampaignsInPeriod` counts campaigns *created* in the active billing period, whether or not they are still active.
- **Which plan's limit applies:** `SubscriptionService.resolveEffectiveCampaignLimit` (lines 734-748).
- **State changes, all in `SubscriptionService.java`:**
  - `requestDowngrade` (442) and `cancelDowngrade` (512)
  - `handlePaymentFailed` (316), `handleInvoicePaid` (260), `handleSubscriptionDeleted` (340) and `handleSubscriptionUpdated` (380)
  - `processExpiredTrials` (557), `processExpiredDowngrades` (591) and `renewExpiredFreeBillingPeriods` (669)
  - `enterTermsPending` (800) and `acceptTerms` (836)
- **Ways in:**
  - `SubscriptionPaidController.java`: `POST /subscription/downgrade` and `/downgrade/cancel`.
  - `stripe/StripeWebhookHandler.java`: `invoice.payment_failed`, `customer.subscription.deleted` and `customer.subscription.updated`.
  - `cron/SubscriptionPeriodProcessorCronJob.java`, daily at 4 AM under ShedLock.
- **Notifications:** `SubscriptionService.publishNotification` publishes `SubscriptionNotificationEvent`. `event/SubscriptionNotificationEventListener.java` picks it up after commit and calls `NotificationService`. Email wording is in `src/main/resources/messages_{en,pl}.properties`.
- **Plan data:** `db/changelog/2026/03/22-03-2026-subscription-tables.sql` seeds FREE 2, BUSINESS 5, ENTERPRISE 10. `2026/04/09-04-2026-update-free-plan-campaign-limit.sql` raises FREE to 5.

**Frontend:**
- `frontend/src/app/feature/plan-billing/plan-billing.component.{ts,html}`: usage ring, payment-failed banner, pending-downgrade banner, which downgrade buttons appear.
- `frontend/src/app/feature/plan-billing/downgrade-confirm-dialog.component.{ts,html}`.
- `frontend/src/app/core/subscription/subscription.service.ts`.
- `frontend/src/app/core/api-frozen/hidden-models.ts`: hand-kept types, because the subscription controllers are hidden from the OpenAPI spec.
- `frontend/src/assets/i18n/{en,pl}.json`.

**What happens today on each path**
1. **Voluntary downgrade to Free.** `requestDowngrade` asks Stripe to cancel at period end, sets `DOWNGRADE_PENDING` and keeps the paid limit.
   - At period end Stripe sends `customer.subscription.deleted`. `handleSubscriptionDeleted` moves the company to Free and opens a **new billing period starting now**, so usage restarts at 0.
   - It logs `PAYMENT_EXHAUSTED` and sends the "All payment attempts … have failed" email, which is wrong for a voluntary downgrade.
   - The job fallback `processExpiredDowngrades` looks for billing periods marked `PENDING_DOWNGRADE`. No production code sets that status (only tests do), so in production this job never does anything.
2. **Enterprise to Business.** A Stripe schedule, applied later by `handleSubscriptionUpdated`, which also opens a new period starting now.
3. **Payment fails.** `handlePaymentFailed` always overwrites the status with `PAYMENT_FAILED` and sets `previousPlan = currentPlan`.
   - The paid limit stays in force for as long as Stripe keeps the subscription alive; nothing in our code limits how long.
   - If the company was `DOWNGRADE_PENDING` or `TERMS_PENDING`, that state is lost.
   - On recovery, `handleInvoicePaid` restores `*_ACTIVE` but leaves `targetPlan` set.
4. **Retries exhausted.** Same as path 1: Free, new period, usage back to 0.
5. **Trial expires or is cancelled.** Free, new period, and `previousPlan` is left at ENTERPRISE.
6. **Existing campaigns.** No subscription code touches them. They stay live and keep receiving applications. Deactivating a campaign is refused while it has active applications (`PartnershipOpportunityService.deactivateOpportunity`, 1479-1488).

**Inconsistent or missing rules**
- **Downgrading adds headroom.** Usage restarts on every drop to a lower plan, so a company that created 10 campaigns on Enterprise today can create 5 more on Free straight away.
- **"Active" is not what's counted.** Deactivating or reactivating a campaign (the update path sets `active` from the request, line 1021) has no effect on the limit.
- **`TERMS_PENDING` can grant a paid limit.** The limit falls back to `previousPlan`, but `enterTermsPending` records `previousState`. A Free company with a stale `previousPlan` (after a trial or a downgrade) gets the Enterprise or Business limit for the 38-day grace period.
- **Paid limit with no end date.** `PAYMENT_FAILED` keeps the paid limit with no deadline on our side.
- **A pending downgrade can get stuck.** If a payment fails while the downgrade is pending and later recovers, the frontend still shows the pending banner (because `targetPlanName` is set). Its "cancel" button then fails, because `cancelDowngrade` requires `DOWNGRADE_PENDING`.
- **Backend and frontend disagree on who can downgrade.** The backend allows a downgrade from `PAYMENT_FAILED` (`requestDowngrade` line 474). The frontend offers no downgrade button there (`availableDowngrades`, plan-billing.component.ts:205-219).
- **The confirm dialog is silent about campaigns.** It says "You'll keep your current benefits until then" and nothing about existing campaigns or usage.
- **The campaign-limit dialog text is never used.** The keys under `campaign_limit` in `en.json`/`pl.json` (around line 6576) are referenced nowhere else in `frontend/src` .ts/.html code.
- **Emails are wrong.** The `trial_expired` email says "Free (2 campaigns/month)" but the database says 5. Voluntary downgrades get the "payment exhausted" email.
- **One failed job blocks campaign creation.** `CampaignLimitService` throws `EntityNotFoundException` when no active billing period exists, so if the Free renewal job fails, companies cannot create campaigns.

## Proposed change
**Policy: existing work is never destroyed; new campaigns are limited by what the company is paying for.**

1. **Existing campaigns are kept.** Nothing is deactivated automatically on any drop to a lower plan. Force-deactivating would break collaborations already under way, and the domain already refuses deactivation while applications are active.

2. **Usage is a rolling window, not tied to the billing period.** Usage = campaigns the company created in the last 30 days (`createdTime > now − 1 month`).
   - It never restarts when the plan changes, so downgrading grants nothing extra.
   - Upgrading still takes effect at once, because the limit goes up.
   - It no longer depends on a billing-period row existing, so a failed renewal job cannot block creation.
   - Alternative considered: keep counting per billing period and carry usage into the new period on downgrades. That preserves the "resets on renewal day" behaviour, but it needs extra period bookkeeping on every state change. I recommend the rolling window unless product wants a hard monthly reset. The copy ("in your current billing period") changes either way.
   - A cap on *concurrently active* campaigns was also considered and rejected. It would need rules for the `active` flag and end dates on every edit, and it contradicts the existing "create up to 5 campaigns" copy.

3. **One function decides the limit**, `SubscriptionEntitlements.effectiveLimit(sub, now)`:

   | status | limit |
   |---|---|
   | `*_ACTIVE`, `TRIAL_ENTERPRISE` | current plan |
   | `DOWNGRADE_PENDING` | current (still paid) plan until period end |
   | `PAYMENT_FAILED` | previous paid plan until a new `payment_grace_deadline` (e.g. 14 days, configurable), then the FREE limit |
   | `TERMS_PENDING` | the limit of `previousState`, resolved recursively, never `previousPlan` |
   | `SUSPENDED_LEGAL`, `ACCOUNT_DEACTIVATED` | 0 |

4. **State changes keep their intent:**
   - A payment failure records `previousState`, doesn't wipe `DOWNGRADE_PENDING`, and on recovery restores `previousState`.
   - A subscription deleted while `DOWNGRADE_PENDING` with a Free target counts as a voluntary downgrade: `SUBSCRIPTION_DOWNGRADED` event and email.
   - A downgrade from `PAYMENT_FAILED` to Free cancels in Stripe immediately, since nothing was paid for the rest of the period.
   - The dead downgrade job becomes a check against Stripe instead of a separate code path that applies the downgrade on its own.

5. **Tell the company, and show the numbers.** The status response gains:
   - `campaignsUsedInWindow`
   - `overLimit`
   - `nextCampaignSlotAt`
   - `limitAfterDowngrade`
   - `paymentGraceDeadline`

   The downgrade dialog warns when current usage is above the target plan's limit. The plan page shows an over-limit banner. The campaign form turns the `CAMPAIGN_LIMIT` error into the existing campaign-limit dialog.

## Plan
1. **Single limit and usage calculation (no behaviour change yet).**
   - Add `subscription/SubscriptionEntitlements.java` with `effectiveLimit` and `usage`, starting as a port of today's `resolveEffectiveCampaignLimit`.
   - `CampaignLimitService` and `SubscriptionService.getStatus` both call it.
   - Unit tests pin today's behaviour first.
   - Files: `CampaignLimitService.java`, `SubscriptionService.java`, `unit/service/subscription/CampaignLimitServiceUnitTest.java`.
2. **Rolling usage.**
   - Add `countCampaignsCreatedSince(userId, since)` to `BillingPeriodRepository`, or better to `PartnershipOpportunityRepository`.
   - In `CampaignLimitService`, stop requiring an active period, but keep a per-company lock for concurrent creation: take a pessimistic lock on the `CompanySubscription` row instead of on `BillingPeriod`.
   - Files: `CampaignLimitService.java`, `repository/CompanySubscriptionRepository.java` (add `findByUserIdForUpdate`), the repository above.
3. **Fix the `TERMS_PENDING` limit.** Resolve from `previousState` in `SubscriptionEntitlements`. File: `SubscriptionService.java`.
4. **Payment-failure grace.**
   - Liquibase changeset adding `company_subscription.payment_grace_deadline`, plus the field on `entity/CompanySubscription.java`.
   - `handlePaymentFailed`: set the deadline only on the first failure; keep `previousPlan` and `previousState` if already `PAYMENT_FAILED`; don't overwrite `DOWNGRADE_PENDING` or `TERMS_PENDING` (record it in `previousState`).
   - `handleInvoicePaid`: restore `previousState` and clear the deadline.
   - Add a property under `app.payments` in `config/AppPaymentsProperties.java`.
5. **Voluntary downgrade vs. retries exhausted.**
   - `handleSubscriptionDeleted`: branch on `status == DOWNGRADE_PENDING && targetPlan == FREE`, logging `SUBSCRIPTION_DOWNGRADED` with `NotificationType.SUBSCRIPTION_DOWNGRADED`.
   - `requestDowngrade` from `PAYMENT_FAILED` to Free: cancel immediately via `stripeService.cancelSubscriptionImmediately`.
   - `processExpiredDowngrades`: either delete it and the unused `PENDING_DOWNGRADE` query, or rewrite it to select `DOWNGRADE_PENDING` subscriptions past period end and confirm with `stripeService.retrieveSubscription` before applying.
   - Files: `SubscriptionService.java`, `BillingPeriodRepository.java`, `cron/SubscriptionPeriodProcessorCronJob.java`.
6. **Status response and warning emails.**
   - Add the fields above to `dto/SubscriptionStatusDtoOut.java`.
   - Notify when a drop to a lower plan leaves `overLimit` true (new `NotificationType` value and email keys).
   - Correct the `trial_expired` email in `messages_{en,pl}.properties` so it doesn't hard-code the number, and the `downgraded` wording.
7. **Frontend types.** Add the new fields to `frontend/src/app/core/api-frozen/hidden-models.ts` and `frontend/src/testing/contract/subscription.contract.ts`, and update the fixtures (`sandbox/fixtures/plan-billing.fixture.ts`, `core/demo/demo-fixtures.ts`). Deploy the backend first; the new fields are optional, so the old frontend keeps working.
8. **Plan and billing screen.**
   - `plan-billing.component.ts`/`.html`: over-limit banner with `nextCampaignSlotAt`; payment-grace deadline in the payment-failed banner; offer a Free downgrade when `PAYMENT_FAILED`; show the pending-downgrade banner only when `status === DOWNGRADE_PENDING`, not just when `targetPlanName` is set.
   - `downgrade-confirm-dialog.component.{ts,html}`: pass `campaignsUsedInWindow` and `limitAfterDowngrade`, and show a "your existing campaigns stay live; new campaigns are blocked until …" warning.
   - i18n keys in `en.json`/`pl.json`, including the "in your current billing period" wording.
9. **Campaign form.** In `frontend/src/app/feature/opportunities/opportunity-form.component.ts`, map the backend `CAMPAIGN_LIMIT` error to a dialog built on the unused `campaign_limit` i18n keys, with an upgrade link. First confirm the HTTP status and error-code shape the backend returns (see Evidence).

## Risks and invariants
- **Existing campaigns are never deactivated by a subscription change.**
  - Test: `SubscriptionService_Downgrade_IntegrationTest` asserts that `active` is unchanged after `handleSubscriptionDeleted` or `processExpiredTrials`.
- **A drop to a lower plan never raises headroom:** usage right after the drop equals usage just before it.
  - Tests: new cases in `CampaignLimitServiceUnitTest` for Enterprise→Free (webhook), trial expiry and trial cancel.
- **The limit never exceeds what is paid for:** `TERMS_PENDING` resolves via `previousState`, and `PAYMENT_FAILED` past its deadline gets the FREE limit.
  - Tests: `SubscriptionService_TermsUnitTest`, `SubscriptionService_WebhookUnitTest`.
- **Webhooks stay idempotent and state-preserving.** A duplicate `invoice.payment_failed` must not move the grace deadline or overwrite `previousState`. The existing event-id unique constraint and `@Version` stay in place.
  - Tests: `StripeWebhookHandlerUnitTest`, `SubscriptionService_Webhook_IntegrationTest`.
- **The job and the webhook can't both apply a downgrade.** Each checks status under a lock; the job asks Stripe before acting.
  - Test: `SubscriptionService_Cron_IntegrationTest`, rewritten to stop setting the unused `PENDING_DOWNGRADE` by hand.
- **Concurrent campaign creation.** Moving the lock from `BillingPeriod` to `CompanySubscription` must still serialise two creates for the same company.
  - Test: add a concurrency integration test.
- **Payments switched off** (`app.payments.enabled=false`): the limit calculation and rolling usage must work for Free companies with no Stripe.
  - Tests: `SubscriptionService_PaymentsDisabled_IntegrationTest`, `PaymentsDisabledBootGuardUnitTest`.
- **Semantics change for Free users.** They lose the renewal-day reset. This is a product decision to confirm before step 2; after the change the landing and billing copy must match.
- **Frontend checks:** `plan-billing.component.spec.ts` and `downgrade-confirm-dialog.component.spec.ts` for the banners and warnings; `subscription.service.spec.ts` for the types.

## Evidence
- FACT: the limit counts campaigns created between the active period's start and end, regardless of `active` (`BillingPeriodRepository.java:28-33`, `CampaignLimitService.java:22-37`).
- FACT: `CampaignLimitService.enforceLimit` is only called on the create path (`PartnershipOpportunityService.java:806-808`; Grep of `src/main` shows no other caller besides the test controller's injection).
- FACT: `handleSubscriptionDeleted`, `processExpiredTrials`, the trial-cancel branch of `requestDowngrade`, `handleSubscriptionUpdated` and `processExpiredDowngrades` each close the active period and open a new one starting now (`SubscriptionService.java` 355-371, 564-575, 451-463, 409-424, 602-613).
- FACT: `handleSubscriptionDeleted` always logs `PAYMENT_EXHAUSTED` and sends `SUBSCRIPTION_PAYMENT_EXHAUSTED` (372-376). The email text says all payment attempts failed (`messages_en.properties:666`).
- FACT: no production code sets `BillingPeriodStatus.PENDING_DOWNGRADE`; only tests and the query at `BillingPeriodRepository.java:25` mention it (Grep across `backend`).
- FACT: `handlePaymentFailed` unconditionally sets `previousPlan = currentPlan` and `PAYMENT_FAILED` (327-328). `handleInvoicePaid` restores `resolveActiveStatus(previousPlan)` without clearing `targetPlan` (275-278).
- FACT: `resolveEffectiveCampaignLimit` uses `previousPlan` for `DOWNGRADE_PENDING`, `PAYMENT_FAILED` and `TERMS_PENDING` (734-748). `enterTermsPending` sets `previousState`, not `previousPlan` (814-817). Free paths leave `previousPlan` set to the paid plan (361, 569, 605).
- FACT: `cancelDowngrade` requires `DOWNGRADE_PENDING` (517-519). The frontend shows the cancel banner whenever `targetPlanName` is set (`plan-billing.component.html:172-212`).
- FACT: the backend allows a downgrade from `PAYMENT_FAILED` (472-476); the frontend offers downgrades only from `ENTERPRISE_ACTIVE` and `BUSINESS_ACTIVE` (`plan-billing.component.ts:205-219`).
- FACT: plan limits are seeded 2/5/10 and FREE is raised to 5 (`22-03-2026-subscription-tables.sql:192-195`, `09-04-2026-update-free-plan-campaign-limit.sql:8`). The `trial_expired` email says "2 campaigns/month" (`messages_en.properties:657`).
- FACT: deactivating a campaign is refused while it has active applications (`PartnershipOpportunityService.java:1479-1488`).
- FACT: the downgrade dialog copy mentions only "current benefits" (`en.json:378-379`, `downgrade-confirm-dialog.component.html`).
- FACT: the `campaign_limit` i18n block exists (`en.json:6576`), but a Grep of `frontend/src` .ts/.html found no reference to it or to `CAMPAIGN_LIMIT`.
- FACT: the subscription response shape is hand-maintained in `hidden-models.ts:43-56` because the controllers are `@Hidden` (`SubscriptionPaidController.java:42`, `hidden-models.ts:1-20`).
- FACT: `CampaignLimitService` throws `EntityNotFoundException` when there is no active period (26-27). Free periods are renewed by the daily job (`SubscriptionPeriodProcessorCronJob.java:31-44`).
- FACT: notifications are delivered after commit by `SubscriptionNotificationEventListener` (lines 28-43).
- INFERENCE: after a drop to a lower plan a company can immediately create up to the Free limit again, whatever it created earlier in the paid period. This follows from the new-period-starting-now facts combined with the created-in-period count.
- INFERENCE: a failed payment during a pending downgrade that later recovers leaves the account `*_ACTIVE` with `targetPlan` set, a pending banner shown and cancel failing with `IllegalStateException`. This combines the three `handlePaymentFailed`/`handleInvoicePaid`/`cancelDowngrade` facts above.
- HYPOTHESIS: how long `PAYMENT_FAILED` keeps the paid limit depends on the Stripe dashboard retry settings. If they end in "mark unpaid" rather than "cancel", `customer.subscription.deleted` never arrives and the paid limit lasts indefinitely. Check the Stripe account's subscription retry and dunning settings.
- HYPOTHESIS: `BusinessRuleViolationException` maps to a specific HTTP status and error-code field that the campaign form can recognise. Check its handler under `common/exceptions` (for example `BaseExceptionHandler`) before step 9.
- HYPOTHESIS: `opportunity-form.component.ts` currently shows a generic error on a limit rejection. Read its submit error handler to confirm.
- HYPOTHESIS: product wants the rolling window rather than the renewal-day reset. The product owner has to decide before step 2.
