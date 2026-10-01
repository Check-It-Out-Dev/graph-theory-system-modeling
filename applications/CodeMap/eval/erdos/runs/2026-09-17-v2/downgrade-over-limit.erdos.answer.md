# How downgrades and failed payments affect campaign limits today, and one policy to fix it

## Problem
A company with more campaigns than the Free plan allows either downgrades or falls back to Free after a failed payment. I traced what the backend and frontend do in that case and where the rules disagree or are missing, and propose one policy with an ordered plan. This only counts as done if the plan limit comes from one place, every state change applies it the same way, the user is told, and the plan and billing screens show what the backend enforces.

The biggest finding: **the backend has no rule about *active* campaigns at all.** The "campaign limit" is a cap on how many campaigns a company can create per billing period, and moving to a lower plan never touches existing campaigns.

## Where it lives today
**Backend, subsystem [11] Subscriptions, payments & consent (`backend/src/main/java/com/sm/instagram/platform/subscription/`)**
- `CampaignLimitService.java`: `enforceLimit` locks the active billing period, counts the campaigns created in it and throws `CampaignLimitExceededException` when the count reaches the limit. Its only production caller is `PartnershipOpportunityService.saveFromDto` in [4] (line 807), when a campaign is created.
- `SubscriptionService.java` (948 lines) holds every state change:
  - `resolveEffectiveCampaignLimit` works out the limit for the current state.
  - `requestDowngrade` and `cancelDowngrade` handle downgrades.
  - The Stripe webhook handlers are `handlePaymentFailed`, `handleInvoicePaid`, `handleSubscriptionDeleted` and `handleSubscriptionUpdated`.
  - The daily cron work is `processExpiredTrials`, `processExpiredDowngrades` and `renewExpiredFreeBillingPeriods`.
  - Terms re-acceptance is `enterTermsPending` and `acceptTerms`.
- `repository/BillingPeriodRepository.java`: `countCampaignsInPeriod` and `findExpiredPendingDowngrades`.
- `stripe/StripeWebhookHandler.java` maps Stripe events to those handlers. `cron/SubscriptionPeriodProcessorCronJob.java` runs at 04:00 daily.
- `event/SubscriptionNotificationEventListener.java` turns events into notifications after commit. Emails go through `NotificationEmailService` in [12].
- Plan data: `db/changelog/2026/03/22-03-2026-subscription-tables.sql` seeds FREE=2, BUSINESS=5, ENTERPRISE=10. `db/changelog/2026/04/09-04-2026-update-free-plan-campaign-limit.sql` raises FREE to 5.

**Frontend, subsystem [174] FE plan, billing & consent**
- `frontend/src/app/feature/plan-billing/plan-billing.component.ts`: the plan cards, downgrade options and usage ring.
- `frontend/src/app/feature/plan-billing/downgrade-confirm-dialog.component.ts` and its `.html` template.
- `frontend/src/app/core/subscription/subscription.service.ts`, which wraps the client in `core/api-frozen/`.

**How the flows work today**
- **Scheduled downgrade to Free:** `requestDowngrade` asks Stripe to cancel at period end and sets `DOWNGRADE_PENDING`. At period end Stripe sends `customer.subscription.deleted`, and `handleSubscriptionDeleted` moves the company to `FREE_ACTIVE` with a fresh Free billing period. The cron path for this never matches (see finding 2).
- **Enterprise to Business:** a Stripe schedule is created. When it releases, Stripe sends `customer.subscription.updated`, and `handleSubscriptionUpdated` switches the plan, starts a new period and sends `SUBSCRIPTION_DOWNGRADED`.
- **Failed payment:** `invoice.payment_failed` leads to `handlePaymentFailed`, which sets `PAYMENT_FAILED` and keeps the plan. Either `invoice.paid` brings the company back, or `customer.subscription.deleted` ends in Free with `PAYMENT_EXHAUSTED`.
- **Existing campaigns** are never touched by any of these paths.

**Where the rules disagree or are missing**
1. **Limit shown vs limit enforced.** The Free card hardcodes a limit of 2 (`plan-billing.component.ts:72-106`, "limits the landing tiers (2/5/10)"). The database has said 5 since the April changeset, so Free and Business are both 5 and a downgrade from Business changes nothing, but the page shows 2.
2. **The downgrade cron can never match.** `findExpiredPendingDowngrades` looks for billing periods with status `PENDING_DOWNGRADE`. No production code sets that status; only tests do. So:
   - A downgrade with no Stripe subscription id stays `DOWNGRADE_PENDING` forever, because `requestDowngrade` skips Stripe when the id is null.
3. **A voluntary downgrade is reported as a failed payment.** The real path is `handleSubscriptionDeleted`. It logs `PAYMENT_EXHAUSTED` and sends a notification saying "All payment attempts for {oldPlan} failed". No `SUBSCRIPTION_DOWNGRADED` notification is sent for a downgrade to Free.
4. **A failed payment keeps paid limits and can lift the cap entirely.**
   - In `PAYMENT_FAILED` the limit stays at the paid plan's (`previousPlan`).
   - `handlePaymentFailed` neither extends nor expires the billing period, so it stays active with an end date in the past.
   - The count only includes campaigns created between the period's start and end, so campaigns created after that end date are never counted. In effect there is no cap until Stripe gives up.
5. **A failed payment during a pending downgrade loses the downgrade.**
   - `handlePaymentFailed` overwrites `DOWNGRADE_PENDING`.
   - After recovery, `handleInvoicePaid` restores an active status but leaves `targetPlan` set, while Stripe still has cancel-at-period-end.
   - The page then shows an active badge with no downgrade options, because `availableDowngrades` returns nothing when `targetPlanName` is set.
6. **`TERMS_PENDING` uses the wrong plan's limit.** `enterTermsPending` saves `previousState` but not `previousPlan`, yet the limit is read from `previousPlan`. After checkout `previousPlan` is the old plan (for example FREE), so a Business company waiting to accept new terms gets Free's limit. A company that never changed plan (`previousPlan` null) keeps its current plan's limit.
7. **Moving down mid-period resets the count.** Cancelling a trial, a trial expiring and payment exhaustion all expire the old period and start a new one from now. The count starts at zero, so a company can create 10 campaigns on a trial, cancel it, and create 5 more straight away.
8. **A missing billing period breaks campaign creation.** `enforceLimit` throws `EntityNotFoundException` when there is no active period. Only `FREE_ACTIVE` periods are renewed by the cron.
9. **The count ignores campaign state.** It includes inactive and finished campaigns. "Active campaigns" is not a concept anywhere in the limit logic.
10. **The frontend never explains a refusal.** An i18n block `campaign_limit` exists in `pl.json` (line 6576), but no non-spec file under `frontend/src/app` uses that key or the `CAMPAIGN_LIMIT` code. The downgrade dialog says nothing about campaigns.

## Proposed change
**Policy: existing campaigns are never stopped; the plan limit only controls new campaigns, counted over a rolling 30-day window.**
- **Existing campaigns keep running.** A downgrade or failed payment never deactivates a campaign. Influencers may have applications in progress, and `deactivateOpportunity` already refuses to deactivate a campaign with active applications (`PartnershipOpportunityService.java:1479-1486`).
- **New campaigns:** a company can create one only if it created fewer than its effective limit in the last month, counted as `[now - 1 month, now]`. The window no longer depends on billing-period boundaries, which fixes findings 4, 7 and 8 at once.
- **Effective limit, defined in one place:**

  | State | Limit |
  |---|---|
  | `FREE_ACTIVE`, `BUSINESS_ACTIVE`, `ENTERPRISE_ACTIVE` | the current plan's |
  | `TRIAL_ENTERPRISE` | Enterprise |
  | `DOWNGRADE_PENDING` | the current plan's (already paid for until period end) |
  | `PAYMENT_FAILED` | Free (no paid-level creation while unpaid) |
  | `TERMS_PENDING` | the limit of the state it came from (`previousState`), never `previousPlan` |
  | `SUSPENDED_LEGAL`, `ACCOUNT_DEACTIVATED` | 0 |

- **Over the limit is a visible state, not an error.** The status response gains `liveCampaigns` (active and not yet ended) and `overLimit`. The downgrade dialog and the notification sent when a company moves to a lower limit say which campaigns keep running and how many new ones are allowed.

**The alternative, and why I didn't pick it:** cap *live* campaigns and pause the extra ones on downgrade. It matches the wording "more active campaigns than the plan allows". But it would break collaborations already under way, clash with the existing refusal to deactivate campaigns that have applications, and contradict what users are already told: "used {{used}} of {{limit}} campaigns in the current billing period" and "You'll keep your current benefits until then". Product should confirm the choice (see Risks).

## Plan
Each step leaves the system working.

1. **One limit policy on the backend** (no API change yet).
   - `subscription/CampaignLimitService.java`: add `effectiveLimit(CompanySubscription)` using the table above and `usage(userId, now)`. Remove the billing-period lock and the `EntityNotFoundException`, and lock the `CompanySubscription` row instead so concurrent creations can't both pass.
   - `repository/BillingPeriodRepository.java` or `PartnershipOpportunityRepository`: add `countCampaignsCreatedBetween(userId, from, to)` and `countLiveCampaigns(userId, now)`.
   - `SubscriptionService.java`: `getStatus` and `resolveEffectiveCampaignLimit` delegate to the policy; `TestSubscriptionController.java:172` switches to the new count.
   - Tests: `CampaignLimitServiceUnitTest.java`, plus a new table-driven test over every `SubscriptionStatus`.
2. **Make failed payments keep the state they interrupted.**
   - `SubscriptionService.handlePaymentFailed`: save `previousState` and don't overwrite `previousPlan` when already `PAYMENT_FAILED`.
   - `handleInvoicePaid`: restore `previousState` (so `DOWNGRADE_PENDING` and `targetPlan` survive) and clear `previousState`.
   - `requestDowngrade`: from `PAYMENT_FAILED`, keep the failure recorded in `previousState`.
   - Check the database: `company_subscription_previous_state_check` already allows `PAYMENT_FAILED` and `DOWNGRADE_PENDING`, so no migration is needed.
   - Tests: `SubscriptionService_WebhookUnitTest`, `SubscriptionService_Webhook_IntegrationTest`.
3. **Tell a voluntary downgrade apart from payment exhaustion.**
   - `SubscriptionService.handleSubscriptionDeleted`: if the status was `DOWNGRADE_PENDING` with target Free, log `SUBSCRIPTION_DOWNGRADED` and send `SUBSCRIPTION_DOWNGRADED`; otherwise keep `PAYMENT_EXHAUSTED`.
   - Replace `processExpiredDowngrades` (dead) with a reconciliation over `CompanySubscription` rows in `DOWNGRADE_PENDING` whose active period has ended. It applies the target plan only for rows without a `stripeSubscriptionId`, and logs a warning otherwise (the webhook owns that case).
   - `CompanySubscriptionRepository.java`: add the query. Remove `findExpiredPendingDowngrades` and the `PENDING_DOWNGRADE` fixtures in `SubscriptionService_CronUnitTest.java`, `SubscriptionService_Cron_IntegrationTest.java` and `SubscriptionRepository_Query_IntegrationTest.java`. Keep the enum value and CHECK constraint to avoid a migration.
4. **Over-limit notification.**
   - In every path that lowers the limit (`handleSubscriptionDeleted`, `handleSubscriptionUpdated` on downgrade, `processExpiredTrials`, trial cancel in `requestDowngrade`, `handlePaymentFailed`), add `liveCampaigns` and `newLimit` to the notification parameters.
   - New Liquibase changeset under `db/changelog/2026/09/` with en/pl message text for the downgraded, payment-failed, payment-exhausted and trial-expired notifications that mentions campaigns ("your N live campaigns keep running…").
   - Templates: `NotificationEmailService` (mapping at lines 87-92).
5. **Additive API fields.**
   - `dto/SubscriptionStatusDtoOut.java`: add `liveCampaigns`, `overLimit`, `campaignWindowStart`, and `planLimits: {FREE, BUSINESS, ENTERPRISE}` read from `subscription_plan`.
   - Regenerate the OpenAPI spec. Update `frontend/src/app/core/api-frozen/hidden-models.ts` and `frontend/src/testing/contract/subscription.contract.ts`.
6. **Frontend.**
   - `plan-billing.component.ts`: take the limits from `planLimits` and delete the hardcoded 2/5/10. The usage label becomes "last 30 days", with an over-limit banner when `overLimit` is true (`plan-billing.component.html` around line 128).
   - `downgrade-confirm-dialog.component.ts` and `.html`: pass `liveCampaigns` and the target limit, and show the keep-running sentence.
   - The campaign-creation screen in [171] (probably `opportunity-form.component.ts`, not read) maps the `CAMPAIGN_LIMIT` refusal to the existing `campaign_limit` i18n block with a link to plan and billing.
   - Update fixtures `plan-billing.fixture.ts`, `downgrade-confirm.fixture.ts` and `demo-fixtures.ts`, and the i18n files `en.json` and `pl.json`.
7. **Cleanup:** remove the now-unused `billingPeriodRepo` lock path from `CampaignLimitService`. Update the demo caption in `pl.json:2908` that describes the old cron behaviour.

## Risks and invariants
- **Existing campaigns are never deactivated by a billing change.** Guard: an integration test in `SubscriptionService_Downgrade_IntegrationTest` asserting live campaigns are unchanged after a downgrade, exhaustion, or trial expiry.
- **The count can't be bypassed by concurrent requests.**
  - Today's pessimistic lock is on the billing period. Moving it to `CompanySubscription` (the `@Version` column plus `PESSIMISTIC_WRITE`) keeps two parallel creations from both passing.
  - Guard: a concurrency test in `CampaignLimitServiceUnitTest` and an integration test.
- **Webhooks stay idempotent.** The unique `stripe_event_id` in `StripeWebhookHandler` stays the guard. The new branch in `handleSubscriptionDeleted` must not send two notifications on retry. Guard: `StripeWebhookHandlerUnitTest` replaying the same event.
- **The Stripe state and our database agree.**
  - Keeping `DOWNGRADE_PENDING` through a failed payment matches Stripe's cancel-at-period-end.
  - Guard: a new webhook test covering downgrade requested, then payment failed, then invoice paid, then subscription deleted, ending in `FREE_ACTIVE` with one `SUBSCRIPTION_DOWNGRADED` event.
- **Stricter limit while a payment is failing.**
  - Companies in `PAYMENT_FAILED` drop to Free's creation limit during Stripe's retries. That is a product decision; if product prefers leniency, change one row of the table.
  - The frontend banner in `payment_failed` (`en.json:345`) must say so.
- **The rolling window changes what the usage ring shows around renewal.** Tests in `plan-billing.component.spec.ts` and the subscription contract test must be updated together with the regenerated models.
- **The terms flow.**
  - Using `previousState` for `TERMS_PENDING` changes limits for companies currently in that state. Before deploying, query production for `status='TERMS_PENDING'`.
  - Guard: `SubscriptionService_TermsUnitTest`.
- **Payments switched off.** `SubscriptionPeriodProcessorCronJob` only runs the paid branches when payments are enabled. The new reconciliation must stay behind the same check. Guard: `SubscriptionService_PaymentsDisabled_IntegrationTest`.
- **Open questions:**
  - Product must confirm "keep running, limit only new campaigns" against the live-cap alternative.
  - Ops must confirm what Stripe does after the last retry (cancel, or mark the subscription unpaid). If it marks it unpaid, no `customer.subscription.deleted` arrives and companies stay in `PAYMENT_FAILED` indefinitely.

## Evidence
- FACT: `enforceLimit` counts campaigns in the active billing period and throws `EntityNotFoundException` when there is none (`subscription/CampaignLimitService.java:22-37`).
- FACT: `countCampaignsInPeriod` counts every campaign by `createdTime BETWEEN start AND end`, with no filter on active or end date (`repository/BillingPeriodRepository.java:28-33`).
- FACT: the only production call site is when a campaign is created (`PartnershipOpportunityService.java:806-808`). The graph also shows `CALLS`/`INJECTS` edges from `PartnershipOpportunityService` and `TestSubscriptionController` only. Updating a campaign sets `active` from the request with no limit check (line 1021).
- FACT: `resolveEffectiveCampaignLimit` uses `previousPlan` for `DOWNGRADE_PENDING`, `PAYMENT_FAILED` and `TERMS_PENDING`, and 0 for `SUSPENDED_LEGAL` (`SubscriptionService.java:734-748`).
- FACT: `enterTermsPending` saves `previousState` but not `previousPlan` (`SubscriptionService.java:800-829`), and `handleCheckoutCompleted` sets `previousPlan` to the old plan (lines 238-239).
- FACT: `handlePaymentFailed` sets `previousPlan = currentPlan` and `PAYMENT_FAILED` and doesn't touch the billing period (lines 316-337). `handleInvoicePaid` restores the active status but not `targetPlan` (lines 275-284).
- FACT: `handleSubscriptionDeleted` always logs `PAYMENT_EXHAUSTED` and sends `SUBSCRIPTION_PAYMENT_EXHAUSTED` (lines 372-376). The message text is "All payment attempts for {oldPlan} failed…" (`23-03-2026-subscription-notification-translations.sql:38`).
- FACT: `requestDowngrade` skips Stripe when `stripeSubscriptionId` is null and still sets `DOWNGRADE_PENDING` (lines 483-501). Trial cancel expires the period and starts a new one from now (lines 447-469). `processExpiredTrials` does the same (lines 557-588).
- FACT: `PENDING_DOWNGRADE` appears in backend production code only in the enum, the repository query, the SQL CHECK constraint and docs. It is set only in tests (search over `backend/`).
- FACT: the cron runs `processExpiredTrials` and `processExpiredDowngrades` only when payments are enabled, and always runs `renewExpiredFreeBillingPeriods` (`cron/SubscriptionPeriodProcessorCronJob.java:31-45`). The renewal only handles `FREE_ACTIVE` (`SubscriptionService.java:669-702`).
- FACT: the seed data sets FREE=2, BUSINESS=5, ENTERPRISE=10 (`22-03-2026-subscription-tables.sql:192-195`), and a later changeset sets FREE to 5 (`09-04-2026-update-free-plan-campaign-limit.sql:8`). No other `campaign_limit` update exists under `backend/src/main`.
- FACT: the frontend hardcodes the Free limit as 2 (`plan-billing.component.ts:72-78`). Downgrade options are hidden when `targetPlanName` is set (lines 205-219). The usage ring uses the backend's `campaignLimit` and `campaignsUsedThisPeriod` (lines 296-301).
- FACT: the downgrade dialog only calls `requestDowngrade` and shows plan and date text with nothing about campaigns (`downgrade-confirm-dialog.component.ts` and `.html`). The English text is "You'll keep your current benefits until then" (`en.json:379`).
- FACT: no non-spec file under `frontend/src/app` references `campaign_limit` or `CAMPAIGN_LIMIT` (search).
- FACT: deactivating a campaign is refused when it has active applications (`PartnershipOpportunityService.java:1479-1486`).
- FACT: webhook idempotency comes from the Stripe event id check plus the database unique constraint, with the optimistic-lock error rethrown (`stripe/StripeWebhookHandler.java:41-60`). Notifications are sent after commit (`SubscriptionNotificationEventListener.java:28-43`).
- INFERENCE: in `PAYMENT_FAILED` after the period end date, new campaigns aren't counted, so there is no cap. The count window is `[start, end]` in the past, and nothing extends or expires the period on a failed payment.
- INFERENCE: a Free downgrade only completes through Stripe's `customer.subscription.deleted` webhook, because the cron query can't match any production row.
- HYPOTHESIS: `CampaignLimitExceededException` (a `BusinessRuleViolationException`) reaches the frontend as a 4xx carrying `CAMPAIGN_LIMIT`. To check, read the exception handler in [7].
- HYPOTHESIS: the campaign form is `opportunity-form.component.ts` in [171] and shows only a generic error. To check, read it.
- HYPOTHESIS: Stripe is set to cancel the subscription after the last failed retry. To check, look at the Stripe dashboard's retry settings.
