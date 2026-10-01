# Step-up authentication for billing: how it works today and a plan to cover every money action

## Problem
Every action that moves money or changes billing should require a step-up token: plan upgrades and downgrades, cancellations, payment-method changes and the company data that goes on invoices. It's done when each such endpoint rejects a request that has no valid single-use token scoped to billing, the frontend asks for the step-up code before each of those calls, and tests prove both.

**Today none of these endpoints check a step-up token.** Only email change and password change do.

## Where it lives today

**Backend step-up core** (subsystems [9] and [6], `backend/src/main/java/com/sm/instagram/platform/auth/stepup/`):
- **`StepUpActionType.java`** has only `EMAIL_CHANGE` and `PASSWORD_CHANGE`.
- **`StepUpChallengeType.java`** has `EMAIL_CODE` and `TOTP`.
- **`StepUpAuthController.java`**, under `/step-up`, requires a signed-in user and has a strict rate limit. It has three calls:
  - `GET /check?actionType=`
  - `POST /request`: sends an email code, or does nothing for TOTP.
  - `POST /verify`: admins verify with TOTP, everyone else with the email code.
- **`StepUpAuthService.java`**:
  - `checkRequirement`: `PENDING_ADMIN` is refused, `ADMIN` gets TOTP. A company or influencer whose initial account setup is incomplete gets "not required"; everyone else gets an email code.
  - The code is 6 digits, stored as a SHA-256 hash for 10 minutes and sent to the user's *last verified* email. Five wrong tries start a 15-minute cooldown; three cooldowns lock the action for 24 hours.
  - The token is a UUID kept in Redis for 10 minutes. The Redis key is `hash(uid):ACTION`, so a token only works for the action it was issued for.
  - `validateAndConsumeToken` reads and deletes the token in one atomic step (single use). A failure throws `AuthenticationTranslatableException`, which the global exception handler turns into **401**.
  - `validateTokenIfRequired` skips the check whenever `checkRequirement` says it isn't required.

**Endpoints that enforce it today** (all read the `X-Step-Up-Token` header):
- `UserController` PATCH and PUT `/{id}`, only when the email actually changes (`EMAIL_CHANGE`).
- `FirebaseAuthProxyController` password change (`PASSWORD_CHANGE`).
- `WebSecurityConfiguration` already lists the header in its allowed headers.

**Billing endpoints with no step-up** (subsystem [11]):

| Endpoint | File | What it does | Protected? |
|---|---|---|---|
| `POST /subscription/upgrade` | `subscription/SubscriptionPaidController.java` | First purchase goes to Stripe Checkout. With an existing subscription it **changes the Stripe price in place, with immediate proration** (`SubscriptionService.initiateUpgrade`, lines 162–181) | No |
| `POST /subscription/downgrade` | same | Downgrade to FREE cancels at period end; on a trial it is an immediate trial cancellation (`requestDowngrade`, lines 447–464) | No |
| `POST /subscription/downgrade/cancel` | same | Brings back the paid renewal | No |
| `POST /subscription/portal` | same | Opens the Stripe Customer Portal, where payment methods are changed and subscriptions cancelled. `StripeService.createPortalSession` passes no portal configuration | No |
| `POST /subscription/trial/activate` | same | FREE → TRIAL_ENTERPRISE | No |
| `POST /subscription/consent` | same | Records the consent proof (not a money action) | No |
| `DELETE /registry/company-data`, `POST /registry/refresh`, `POST /registry/confirm` | `registry/RegistryController.java` ([11]) | Change `CompanyData`: company name, NIP (Polish tax ID) and address. `InvoiceRetryService` reads these **when it retries an invoice** (lines 78–92), so a change here alters the buyer on Fakturownia invoices | No |

Both subscription controllers require the `COMPANY` role and are `@Hidden` from OpenAPI. `SubscriptionPaidController` is only registered when `app.payments.enabled=true`. Webhooks, the dev event poller, the cron jobs and `TestSubscriptionController` are machine or test paths and must stay ungated.

**Frontend** (subsystems [170], [174], [177]):
- **Step-up plumbing:**
  - `core/step-up/step-up.service.ts` wraps the generated `/step-up` client, including the generated `StepUpActionType`.
  - `core/step-up/step-up-context.ts` defines the `STEP_UP_TOKEN` HttpContext token and `withStepUpToken()`.
  - `core/interceptors/step-up.interceptor.ts` adds the header.
  - `shared/components/step-up-dialog/step-up-dialog.component.ts` calls request and then verify. It closes with the token, or with `null` if cancelled or not required.
- **The one existing caller:** `feature/profile/email-change.component.ts` opens the dialog, then `user.service.ts#patch(id, dto, stepUpToken)` sends the token.
- **Billing screens:**
  - `feature/plan-billing/plan-billing.component.ts`: `activateTrial`, `startUpgrade`, `startDowngrade`, `cancelTrial`, `cancelPendingDowngrade`, `openCustomerPortal`.
  - `upgrade-confirm-dialog.component.ts` calls `recordConsent` and then `initiateUpgrade`.
  - `downgrade-confirm-dialog.component.ts` calls `requestDowngrade`.
  - Both go through `core/subscription/subscription.service.ts` (`SubscriptionWriteApi`) and the hand-written `core/api-frozen/subscription.client.ts`. None of these send a step-up token.
- **Error handling:** `core/interceptors/error.interceptor.ts` treats a 401 on a signed-in session as "refresh the session and retry once". It only clears the session if the refresh itself returns 401.

## Proposed change

**1. Separate action types for billing.** Add three values to `StepUpActionType`:
- `BILLING_PLAN_CHANGE`: upgrade, downgrade, trial cancellation, cancelling a pending downgrade, trial activation.
- `BILLING_PORTAL`: payment methods and cancellation inside Stripe.
- `BILLING_DETAILS_CHANGE`: company and invoice data.

Because the Redis key includes the action, an email-change token can never unlock a payment, and one billing token can't be replayed on another billing endpoint.

**2. No "setup incomplete" shortcut for billing.** For email change, `checkRequirement` returns "not required" when setup is incomplete. Billing actions should always require step-up (TOTP for admins, email code for everyone else). The server must decide this; the dialog keeps handling `required: false` for the existing actions.

**3. Enforce with an annotation, not inline calls (recommended).** Two ways to do it:
- **(a) Inline calls**, the same pattern as `UserController`. Smallest change, but every future billing endpoint has to remember it.
- **(b) A `@RequiresStepUp(action)` annotation with an aspect** that reads `X-Step-Up-Token` and calls `validateAndConsumeToken` before the method body runs. `RecaptchaValidationAspect` / `@RequiresRecaptcha` in [8] already use this pattern. Add a reflection test that fails if a write mapping on the billing controllers has no annotation.

I recommend (b). It is declarative, easy to audit, and the reflection test stops new endpoints from slipping through. Keep the existing 401 and message key `error.stepup.invalid_token` so the dialog and error handling behave the same way.

**4. Company data: gate changes, not first entry.** Gate `DELETE /registry/company-data` and `POST /registry/refresh`. `POST /registry/confirm` returns 409 when company data already exists, so the only way to change stored invoice data is to reset first, and the reset is gated. That keeps onboarding free of extra friction. The one gap: data being entered for the first time could be changed by someone holding a stolen session before anyone confirms it. That is acceptable, because no invoice exists yet.

**5. Frontend: one shared helper.** Add `StepUpGate.run(action, token => call$)`. It opens `StepUpDialogComponent`; if the user cancels (`null`) it does nothing, and otherwise it runs the call with `withStepUpToken(token)`. The billing clients get an optional `stepUpToken` parameter, the same way `UserApiService.patch` does.
- In the upgrade dialog, do step-up *before* `recordConsent`, so consent isn't recorded for an upgrade that is then refused.

## Plan
Each step leaves the system working. The backend only starts refusing requests in step 5, after the frontend already sends tokens.

1. **Backend: new action types and requirement rule.**
   - `StepUpActionType.java`: add the three values.
   - `StepUpAuthService.checkRequirement`: billing actions skip the incomplete-setup shortcut.
   - Optionally, pass the action to `EmailService.sendStepUpCodeEmail` and its template, so the email says what the user is approving.
   - Add i18n keys to `messages_en.properties` / `messages_pl.properties` ([16]).
   - Regenerate the OpenAPI client so the frontend's `api/model/step-up-action-type.ts` gets the values (`openapi:gen`; never edit the generated client by hand).
2. **Backend: `@RequiresStepUp` and the aspect, in log-only mode.**
   - New `auth/stepup/RequiresStepUp.java` and `auth/stepup/StepUpAspect.java`.
   - Behind a new property, e.g. `app.step-up.billing.enforce=false`: while false, the aspect only logs requests that arrive without a token.
   - Annotate `SubscriptionPaidController`: `upgrade`, `downgrade`, `downgrade/cancel` and `trial/activate` get `BILLING_PLAN_CHANGE`; `portal` gets `BILLING_PORTAL`.
   - Annotate `RegistryController`: `DELETE company-data` and `refresh` get `BILLING_DETAILS_CHANGE`.
   - Leave `consent`, `config`, the reads, the webhooks, the poller, the crons and `TestSubscriptionController` unannotated.
3. **Frontend: clients and gate.**
   - `core/api-frozen/subscription.client.ts` and `core/subscription/subscription.service.ts`: optional `stepUpToken`, sent with `{ context: withStepUpToken(t) }`.
   - `registry.service.ts` ([176]): the same for reset and refresh.
   - New `core/step-up/step-up-gate.service.ts`.
4. **Frontend: screens.**
   - `plan-billing.component.ts`: `openCustomerPortal`, `cancelPendingDowngrade` and `activateTrial` go through the gate.
   - `upgrade-confirm-dialog.component.ts`: gate first, then `recordConsent`, then `initiateUpgrade` with the token.
   - `downgrade-confirm-dialog.component.ts`: gate before `requestDowngrade`. This covers `cancelTrial` too.
   - The company-data reset and refresh screens in [176].
   - Each error classifier maps 401, 423 and 429 to step-up messages.
   - Add Transloco keys to en and pl.
   - Update the demo fixtures (`core/demo/demo-fixtures.ts`, `demo.interceptor`) so demo mode still works.
5. **Backend: switch enforcement on.** Set `app.step-up.billing.enforce=true` in all profiles, after the frontend from step 4 is live. Then remove the flag.
6. **Tests** (details in the next section), added alongside each step.

## Risks and invariants
- **A token can only be used once, and only for its action.**
  - The atomic read-and-delete (FACT) already guarantees single use.
  - The aspect must run *before* any Stripe or database side effect. If the business call later fails (for example 409 "no pending downgrade"), the token is already used up and the user must step up again. Accept this and make sure the error message says so.
  - Tests: a unit test for the aspect (missing, wrong and already-used token → 401, with no call to `SubscriptionService`), and a Cucumber scenario "billing token cannot be reused" modelled on `step-up-auth.feature`'s "Step-up token cannot be reused".
- **Tokens don't cross actions.** An `EMAIL_CHANGE` token sent to `/subscription/upgrade` must get 401. Add a Cucumber scenario for it.
- **No billing path is left out.** A reflection unit test lists every `@PostMapping`/`@DeleteMapping` on `SubscriptionPaidController` and `RegistryController` and checks for `@RequiresStepUp`, with an explicit allow-list: `consent`, `lookup`, `confirm`.
- **Payments stay consistent.** Upgrades still wait for webhooks before changing the database (`SubscriptionService` lines 172–176). Gating adds no new state, so webhook handling, the invoice retry job and the lifecycle crons must keep working with no token. Run the existing `StripeWebhookControllerUnitTest`, the cron unit tests and the subscription integration tests unchanged.
- **Consent order.** If `recordConsent` runs and step-up then fails, a consent proof exists for an upgrade that never happened. Fix the order and test it in `upgrade-confirm-dialog.component.spec.ts`: a cancelled step-up means neither `recordConsent` nor `initiateUpgrade` is called.
- **401s trigger a session refresh.** A missing or bad token returns 401, so the error interceptor refreshes the session and repeats the request once. Nothing changes the first time (the aspect refuses before any side effect), so the repeat is safe but wasted, and the error still reaches the component. Assert that the component shows the step-up error and that the user stays signed in (plan-billing and dialog specs, plus an `error.interceptor` spec case).
- **Stripe Portal is a hand-off.** Once the portal URL is issued, what the user can do inside it (cancel, change card) depends on Stripe dashboard settings; our code passes no portal configuration. Gating session creation is the only control point on our side.
- **Rollout order.** Turning enforcement on before the frontend sends tokens breaks billing for everyone. The flag in steps 2 and 5 prevents that. Tests: the frontend specs plus a rollout smoke test with the flag on.
- **Where the code goes.** The code is sent to the last verified email; accounts without one would never receive it. Add a test for a company account with no last verified email.
- **Frontend unit specs to extend:** `plan-billing.component.spec.ts`, `downgrade-confirm-dialog.component.spec.ts`, `upgrade-confirm-dialog.component.spec.ts`, `subscription.service.spec.ts`, `step-up-dialog.component.spec.ts` (new actions), and a new `step-up-gate.service.spec.ts`.
- **Backend Cucumber feature:** a new `billing-step-up.feature` reusing `StepUpAuthSteps`/`StepUpHooks`. Scenarios:
  - COMPANY upgrade with a valid token succeeds.
  - Upgrade, downgrade, cancel-downgrade and portal each return 401 without a token.
  - Incomplete setup still requires step-up.
  - A company-data reset without a token returns 401.
- **Browser e2e:** the `e2e-tests/` directory isn't indexed, so I can't say which browser e2e tests cover billing today. Whoever owns `e2e-tests/` should check.

## Evidence
- FACT: `StepUpActionType` has only `EMAIL_CHANGE` and `PASSWORD_CHANGE`; `StepUpChallengeType` has `EMAIL_CODE` and `TOTP` (both files read in full).
- FACT: `checkRequirement` refuses `PENDING_ADMIN`, gives `ADMIN` TOTP, returns "not required" for a company or influencer with incomplete setup, and otherwise asks for an email code. The code is a 6-digit SHA-256 hash with a 10-minute TTL, sent to `getLastVerifiedEmail()`. Five tries mean a 15-minute cooldown; three cycles mean a 24-hour lockout. The token is a UUID with a 10-minute TTL, the key is `hash(uid):ACTION`, and it is consumed with `getAndDelete` (`auth/stepup/StepUpAuthService.java`, read in full).
- FACT: the controller exposes `/step-up/check`, `/request` and `/verify`, requires a signed-in user, uses the STRICT rate limit, and verifies admins with TOTP (`StepUpAuthController.java`, read in full).
- FACT: `AuthenticationTranslatableException` becomes 401 (`common/exceptions/GlobalDefaultExceptionHandler.java`, lines 97–98).
- FACT: enforcement exists only in `UserController.java` (lines 282–286 and 314–317, `EMAIL_CHANGE` on an actual change) and `FirebaseAuthProxyController.java` (lines 457–458, `PASSWORD_CHANGE`). A grep for `StepUp` / `X-Step-Up-Token` across controllers, filters and config found nothing else apart from the CORS header in `WebSecurityConfiguration.java:282`.
- FACT: `SubscriptionPaidController.java` (read in full) has no step-up check on trial/activate, consent, upgrade, downgrade, downgrade/cancel or portal. It requires `COMPANY` and is only active when `app.payments.enabled` is true. `SubscriptionController.java` has only GET `/status` and `/invoices`.
- FACT: an upgrade with an existing subscription calls `updateSubscriptionPrice` (immediate proration) and waits for webhooks (`SubscriptionService.java`, lines 136–183). Downgrading a trial to FREE is an immediate cancellation (lines 447–464, from grep).
- FACT: `StripeService.createPortalSession` sets only the customer and return URL (`StripeService.java`, lines 161–171).
- FACT: `RegistryController.java` (read in full): confirm, `DELETE company-data` and refresh have rate limits but no step-up. `InvoiceRetryService.java` (lines 78–92, from grep) builds the Fakturownia buyer from `CompanyData` (company name, NIP, registered address).
- FACT: the frontend step-up plumbing (`step-up.service.ts`, `step-up-context.ts`, `step-up.interceptor.ts`, `step-up-dialog.component.ts`, all read) and its only real caller, `email-change.component.ts` with `user.service.ts#patch(…, stepUpToken)` (grep with context).
- FACT: `subscription.client.ts` and `subscription.service.ts` (read in full) send no step-up context. `plan-billing.component.ts` (lines 364–599) and the two confirm dialogs (grep with context) call the write API without a token; the upgrade dialog calls `recordConsent` and then `initiateUpgrade`.
- FACT: `error.interceptor.ts` (lines 17–40 and 82–127) refreshes and retries on an authenticated 401/419, and clears the session only when the refresh returns 401.
- FACT: `step-up-auth.feature` covers only email change, including token reuse, ADMIN TOTP and cooldown (scenario list from grep).
- INFERENCE: the frontend `StepUpActionType` comes from OpenAPI (it is imported from `api/model/step-up-action-type`), so new values need a client regeneration. `StepUpAuthController` isn't `@Hidden`, unlike the subscription controllers.
- INFERENCE: gating the company-data reset is enough to protect changes, because confirm returns 409 when data already exists (per its `@ApiResponse` documentation, not the service body).
- HYPOTHESIS: `RecaptchaValidationAspect` is a usable pattern for the step-up aspect. It is named in the [8] navigator, but I haven't read it.
- HYPOTHESIS: the step-up email template says "email change" and gives no action context. Check `EmailService.sendStepUpCodeEmail` and its Thymeleaf template.
- HYPOTHESIS: `registry.service.ts` ([176]) uses the generated Registry client and can accept `options.context`. Read it before step 3.
- HYPOTHESIS: some company accounts have a null `lastVerifiedEmail`. Check with a database query or `User.java`.
- HYPOTHESIS: the demo interceptor and fixtures simulate `/step-up/*`; they reference step-up per grep, but I haven't read them.
- HYPOTHESIS: which Stripe Portal features (cancellation, payment-method update) are turned on lives in the Stripe dashboard. The Stripe account owner can confirm.
