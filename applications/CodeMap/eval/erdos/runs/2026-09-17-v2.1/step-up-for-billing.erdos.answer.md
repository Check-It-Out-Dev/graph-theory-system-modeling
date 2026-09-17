# Step-up authentication for billing actions: current state and plan

## Problem
Every action that moves money or changes billing should require step-up authentication: plan changes, cancellations, payment-method changes and invoice data. Today only email changes and password changes require it. The work is done when every billing write refuses a request without a valid single-use step-up token for that action, and the frontend collects that token before calling. Tests in both repositories should prove it.

## Where it lives today

**Backend step-up** (`backend/src/main/java/com/sm/instagram/platform/auth/stepup/`)
- **Controller:** `StepUpAuthController.java` exposes `GET /step-up/check`, `POST /step-up/request` and `POST /step-up/verify`. It requires a signed-in user and uses the STRICT rate limit.
- **Challenge types** (`StepUpChallengeType.java`):
  - `TOTP` for `ADMIN` users.
  - `EMAIL_CODE` for COMPANY and INFLUENCER users. A 6-digit code is hashed, stored in Redis for 10 minutes and emailed to `lastVerifiedEmail`.
- **Action types** (`StepUpActionType.java`): only `EMAIL_CHANGE` and `PASSWORD_CHANGE`.
- **Rules** (`StepUpAuthService.java`):
  - `PENDING_ADMIN` users are refused.
  - COMPANY and INFLUENCER users whose initial setup is incomplete get `required=false`.
  - A code allows 5 attempts, then a 15-minute cooldown. Three cooldowns lock the action for 24 hours.
  - Verifying issues a UUID token stored for 10 minutes. Redis keys combine the user and the action.
- **How the token is checked:** `validateTokenIfRequired` calls `validateAndConsumeToken`, which does an atomic Redis `getAndDelete`. On failure it throws `AuthenticationTranslatableException`, which `GlobalDefaultExceptionHandler` turns into a 401.
- **Endpoints that enforce it:** only two, each written inline in the controller:
  - `user/UserController.java` (PATCH and PUT `/user/{id}`), and only when the email really changes.
  - `auth/FirebaseAuthProxyController.java` (password change).
- **CORS:** `common/authorization/WebSecurityConfiguration.java` already allows the `X-Step-Up-Token` header.

**Frontend step-up**
- `frontend/src/app/core/step-up/step-up.service.ts` wraps the generated client.
- `core/step-up/step-up-context.ts` defines the `STEP_UP_TOKEN` HttpContext token and `withStepUpToken()`.
- `core/interceptors/step-up.interceptor.ts` turns that context into the `X-Step-Up-Token` header. It is first in the chain in `app.config.ts`.
- `shared/components/step-up-dialog/step-up-dialog.component.ts` runs request → code entry → verify, and closes with the token (or `null`).
- The only caller is `feature/profile/email-change.component.ts`, which uses `core/user/user.service.ts`.

**Billing endpoints: none of them check step-up**

`backend/src/main/java/com/sm/instagram/platform/subscription/SubscriptionPaidController.java` (COMPANY users only, active only when `app.payments.enabled` is on):

| Endpoint | What it does | Protected today |
|---|---|---|
| `POST /subscription/upgrade` | Starts a Stripe Checkout (money) | no |
| `POST /subscription/downgrade` | Plan change or cancellation; FREE from a trial cancels the trial at once | no |
| `POST /subscription/downgrade/cancel` | Restores the paid plan (money) | no |
| `POST /subscription/trial/activate` | Plan change | no |
| `POST /subscription/portal` | Opens a Stripe Customer Portal session, where the user can change the payment method and cancel | no |
| `POST /subscription/consent` | Records consent, moves no money | no, and does not need it |
| `GET /subscription/config` | Read only | no, and does not need it |

`SubscriptionController.java` only has reads (`/status`, `/invoices`).

Invoice data lives in `backend/src/main/java/com/sm/instagram/platform/registry/RegistryController.java`. `InvoiceRetryService` builds the Fakturownia buyer (name, NIP, address) from `CompanyData`. These endpoints are unprotected:
- `POST /registry/confirm` (first time only; it answers 409 if company data already exists)
- `DELETE /registry/company-data`
- `POST /registry/refresh`

Stripe webhooks (`StripeWebhookController`) are server-to-server and out of scope. A cancellation made inside the Stripe portal comes back through `handleSubscriptionDeleted`, so the only lever in the app is protecting creation of the portal session.

**Frontend billing flows** (`frontend/src/app/feature/plan-billing/`)
- `plan-billing.component.ts`:
  - `activateTrial()` calls the write API after the trial consent dialog.
  - `cancelPendingDowngrade()` and `openCustomerPortal()` call the write API directly.
  - `startUpgrade`, `startDowngrade` and `cancelTrial` open dialogs that make the write.
- `upgrade-confirm-dialog.component.ts` calls `recordConsent`, then `initiateUpgrade`.
- `downgrade-confirm-dialog.component.ts` calls `requestDowngrade`.
- All of these go through `core/subscription/subscription.service.ts` (`SubscriptionWriteApi`) and then the hand-written `core/api-frozen/subscription.client.ts`. That client cannot pass an HttpContext today.
- `core/registry/registry.service.ts` only wraps lookup, confirm and read. No frontend code outside the generated `api/` folder calls reset or refresh.

## Proposed change
1. **New action types, with the rules in one place.**
   - Add `SUBSCRIPTION_CHANGE` (upgrade, downgrade, cancel downgrade, trial activation, trial cancellation).
   - Add `BILLING_PORTAL` (payment method and in-portal cancellation).
   - Add `INVOICE_DATA_CHANGE` (registry confirm, reset, refresh).
   - Put the rules in `StepUpAuthService.checkRequirement`, so that `/check`, `/request` and enforcement always agree.
   - `SUBSCRIPTION_CHANGE` and `BILLING_PORTAL` fail closed: they ignore the "setup incomplete" skip. If the user has no `lastVerifiedEmail`, refuse with 403 and a new key such as `error.stepup.no_verified_channel`.
   - `INVOICE_DATA_CHANGE` keeps the skip, so the first `confirm` during onboarding still works.
2. **Declarative enforcement.** Add `@RequiresStepUp(action)` and a `StepUpEnforcementAspect`, modelled on the existing `@RequiresRecaptcha` / `RecaptchaValidationAspect`, which reads a header before the method runs. The aspect reads `X-Step-Up-Token` and calls `validateTokenIfRequired` before the controller method runs, so a bad token never reaches Stripe or the database.
   - I chose this over inline checks like `UserController` because there are 8 endpoints, and an annotation can be checked by a reflection test.
   - Leave the conditional email-change check in `UserController` as it is.
3. **Token consumed before the work.** Keep the current order: consume the token, then do the work. The downside is that if Stripe fails, the user must verify again. Consuming after success would lose the atomic single-use guarantee.
4. **Frontend: get the token first, then write.**
   - Add a small helper next to the dialog that opens `StepUpDialogComponent` for an action and returns the token (or `null`).
   - Let the write clients accept a token and pass it via `withStepUpToken`, as `user.service.ts` already does.
   - The upgrade dialog gets the token before `recordConsent`, so no consent is recorded for an upgrade the user then abandons.
5. **Error status.** Keep 401 (`error.stepup.invalid_token`) for consistency with email and password changes. Known cost: the frontend error interceptor treats an authenticated 401 as "refresh the session and replay once". The replay is harmless because it is refused again before any side effect, but it costs an extra refresh call. The alternative is a dedicated 403 for these endpoints, which would split the error contract.

## Plan
Each step leaves production working. The frontend ships before backend enforcement.

1. **Backend: action types and rules** (not enforced yet).
   - `auth/stepup/StepUpActionType.java`: add the three values.
   - `StepUpAuthService.java`: add a fail-closed set for `SUBSCRIPTION_CHANGE` and `BILLING_PORTAL`, and the missing-verified-email refusal.
   - Add the new i18n keys to `messages_en.properties` and `messages_pl.properties`.
   - New `StepUpAuthServiceUnitTest`.
2. **Regenerate the frontend client** with `openapi:gen`, so `frontend/src/app/api/model/step-up-action-type.ts` gets the new values. `StepUpAuthController` is published; the subscription controllers are `@Hidden`.
3. **Backend: the mechanism, not applied yet.**
   - New `auth/annotation/RequiresStepUp.java` and `auth/config/StepUpEnforcementAspect.java`.
   - Unit tests for the aspect.
4. **Frontend: send tokens.** Backend endpoints that are not yet guarded ignore the extra header, so this is safe to ship first.
   - `core/api-frozen/subscription.client.ts`: add an optional `options?: { context?: HttpContext }` to `activateTrial`, `initiateUpgrade`, `requestDowngrade`, `cancelDowngrade` and `createPortalSession`.
   - `core/subscription/subscription.service.ts`: `SubscriptionWriteApi` methods accept an optional `stepUpToken`.
   - `core/registry/registry.service.ts`: `confirm(nip, stepUpToken?)`.
   - New helper in `shared/components/step-up-dialog/` (for example `step-up-gate.ts`) that opens the dialog and returns the token.
   - `upgrade-confirm-dialog.component.ts`: token, then `recordConsent`, then `initiateUpgrade`.
   - `downgrade-confirm-dialog.component.ts`: token, then `requestDowngrade` (covers downgrade and trial cancel).
   - `plan-billing.component.ts`: `activateTrial`, `cancelPendingDowngrade` and `openCustomerPortal` get a token first, and their error classifiers map 401 `error.stepup.*` to a "verification expired, try again" key.
   - Company setup confirm passes a token when one is returned.
   - Update the i18n files, the specs, and `core/demo/demo-fixtures.ts` (the demo tour will now meet the dialog).
5. **Backend: turn enforcement on.**
   - `SubscriptionPaidController.java`: `@RequiresStepUp(SUBSCRIPTION_CHANGE)` on `/upgrade`, `/downgrade`, `/downgrade/cancel` and `/trial/activate`; `@RequiresStepUp(BILLING_PORTAL)` on `/portal`.
   - `RegistryController.java`: `@RequiresStepUp(INVOICE_DATA_CHANGE)` on `/confirm`, `DELETE /company-data` and `/refresh`.
   - Update the CORS comment in `WebSecurityConfiguration.java`.
   - Update `RegistryControllerUnitTest` and the Cucumber tests (see Risks).
6. **Optional:** a property such as `app.step-up.billing.enforced`, like the aspect's global recaptcha switch, so enforcement can be rolled back without a redeploy.

## Risks and invariants
- **Every billing write must be guarded.** Add a reflection unit test: every `@PostMapping` or `@DeleteMapping` in `SubscriptionPaidController` (except `/consent` and `/config`) and the write endpoints of `RegistryController` must carry `@RequiresStepUp`. Otherwise a new endpoint can silently skip it.
- **A token only works for its own action.** Redis keys include the action name, so an `EMAIL_CHANGE` token must not unlock `SUBSCRIPTION_CHANGE`. Test this in the aspect test and in the Cucumber tests.
- **Single use.** Reusing a token gives 401. Extend `step-up-auth.feature`, reusing its existing "completes full step-up flow" and reuse scenarios, to cover upgrade and downgrade.
- **A wrong token destroys the valid one.** `getAndDelete` removes the stored token before comparing. A request with a bad or stale token therefore forces the user to verify again. This is acceptable, but the frontend error copy should say so.
- **No side effect without a token.** Integration tests should assert that no Stripe call is made and no `SubscriptionEvent` or `BillingPeriod` changes when the token is missing.
- **Existing tests will break.** `subscription/subscription-e2e.feature` ("activates trial", "requests downgrade to plan", "cancels pending downgrade") must complete step-up first. `RegistryControllerUnitTest` and the registry and activation tests (`account-activation-e2e.feature`) need the same treatment.
- **Onboarding.** `INVOICE_DATA_CHANGE` keeps the setup skip so the first NIP confirmation still works. A test with a setup-incomplete company confirming without a token must still pass.
- **Fail closed for money.** A setup-incomplete company calling `/subscription/upgrade` without a token gets refused. Add a Cucumber scenario for it.
- **Brute-force budget.** Lockout counts are per action, so three new actions triple the attempts per 24 hours (5×3 per action). With 10⁶ possible codes this is negligible, but record the decision.
- **Frontend order and cancel.** In `upgrade-confirm-dialog.component.spec.ts`: cancelling step-up makes no `recordConsent` or `initiateUpgrade` call; on success, both calls carry the context token only where required. Mirror this in `downgrade-confirm-dialog.component.spec.ts` and `plan-billing.component.spec.ts` (trial, cancel downgrade, portal), in `step-up.interceptor.spec.ts` (unchanged), and in `demo-fixtures.spec.ts`.
- **Stripe portal limit.** The protection covers creating the portal session. Anyone holding the returned URL can still act inside Stripe until that session expires; that expiry is set by Stripe, not by the app.

## Evidence
- FACT: challenge types are `EMAIL_CODE` and `TOTP`; action types are only `EMAIL_CHANGE` and `PASSWORD_CHANGE` (`auth/stepup/StepUpChallengeType.java`, `StepUpActionType.java`).
- FACT: ADMIN gets TOTP, PENDING_ADMIN is refused, and setup-incomplete COMPANY/INFLUENCER users get `required=false`; code TTL, attempts, cooldown, lockout and 10-minute token TTL; keys are per user and action; the token is consumed with atomic `getAndDelete` (`backend/.../auth/stepup/StepUpAuthService.java`, read in full).
- FACT: `/step-up/check`, `/request` and `/verify`; ADMIN is verified by TOTP (`backend/.../auth/stepup/StepUpAuthController.java`).
- FACT: only `UserController.java:282-286,314-317` (email actually changing) and `FirebaseAuthProxyController.java:456-458` read `X-Step-Up-Token` (Grep over `backend/src/main`).
- FACT: `AuthenticationTranslatableException` becomes 401 (`common/exceptions/GlobalDefaultExceptionHandler.java:96-98`).
- FACT: `SubscriptionPaidController` has no step-up checks on trial, consent, upgrade, downgrade, cancel downgrade or portal; it is COMPANY-only and gated by `app.payments.enabled` (file read in full).
- FACT: `RegistryController` confirm, reset and refresh have no step-up; confirm answers 409 if data already exists (file read in full).
- FACT: the Fakturownia buyer comes from `CompanyData` (`subscription/invoicing/InvoiceRetryService.java:78-93`).
- FACT: `requestDowngrade(FREE)` from `TRIAL_ENTERPRISE` cancels the trial immediately (`SubscriptionService.java:447-464`).
- FACT: the aspect pattern reads a header and throws before `proceed()` (`auth/config/RecaptchaValidationAspect.java`, `auth/annotation/RequiresRecaptcha.java`).
- FACT: CORS allows `X-Step-Up-Token` (`WebSecurityConfiguration.java:281-282`).
- FACT: frontend token flow runs dialog → token → `withStepUpToken` → interceptor header; the only consumer is email change (`step-up.service.ts`, `step-up-context.ts`, `step-up.interceptor.ts`, `step-up-dialog.component.ts`, `email-change.component.ts:94-133`; Grep for `withStepUpToken`).
- FACT: the billing write clients cannot pass an HttpContext; the upgrade dialog calls `recordConsent` then `initiateUpgrade`; `plan-billing.component.ts` calls trial, cancel downgrade and portal directly (`subscription.client.ts`, `subscription.service.ts`, `upgrade-confirm-dialog.component.ts:94-118`, `downgrade-confirm-dialog.component.ts:88-100`, `plan-billing.component.ts:364-591`).
- FACT: the error interceptor refreshes and replays once on an authenticated 401 or 419 (`frontend/src/app/core/interceptors/error.interceptor.ts`).
- FACT: the demo fixtures answer `/step-up/check|request|verify` by regex (`frontend/src/app/core/demo/demo-fixtures.ts:1463-1484`).
- FACT: `subscription-e2e.feature` calls trial, downgrade and cancel downgrade without step-up (lines 29, 92, 98); `step-up-auth.feature` has "completes full step-up flow" and token-reuse scenarios.
- FACT: `lastVerifiedEmail` can be null, since the test surface sets it to null (`registry/test/TestRegistryController.java:263`).
- INFERENCE: on a replay after a 419, the header is still there, because the step-up interceptor runs before the error interceptor (`app.config.ts:60,83` ordering comment).
- INFERENCE: a setup-incomplete user is roughly an email-unverified user, because `EmailVerificationService.java:240-254` sets `lastVerifiedEmail` and marks setup complete together. This is why billing actions should fail closed rather than keep the skip.
- HYPOTHESIS: the step-up dialog template uses the same copy for every action. Check `step-up-dialog.component.html` for `data.action` and add per-action i18n if it does not.
- HYPOTHESIS: `company-setup.component.ts` can be reached after setup is complete, which would make the confirm token path live. Check its route guards.
- HYPOTHESIS: OpenAPI contract tests (`OpenApiSpecGeneratorTest`, etc.) snapshot the `StepUpActionType` enum and need refreshing after step 1. Run them.
- HYPOTHESIS: `TestSubscriptionController` is limited to test profiles and does not need the guard. Check its `@Profile` or `@ConditionalOnProperty`.
