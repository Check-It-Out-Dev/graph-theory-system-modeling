I've mapped the step-up mechanism and every billing write in both repos. Next I'll write the answer, then check the few assumptions it depends on.

## Problem
Every backend endpoint that moves money or changes billing data has to refuse the request unless it carries a fresh, single-use step-up token. The frontend has to get that token (email code, or TOTP for admins) just before the call. The change is done when each such endpoint rejects a request with a missing or wrong token, accepts a valid one, and a test fails if someone adds a new billing endpoint without the guard.

## Where it lives today

**Backend step-up** (subsystem [9], package `platform/auth/stepup`)
- `StepUpActionType.java` has only `EMAIL_CHANGE` and `PASSWORD_CHANGE`. `StepUpChallengeType.java` has `EMAIL_CODE` and `TOTP`.
- `StepUpAuthController.java` exposes three calls under `/step-up` (strict rate limit, logged-in users only):
  - `GET /check?actionType=`
  - `POST /request`: sends an email code; for admins it sends nothing because they use TOTP.
  - `POST /verify`: admins are checked with TOTP, everyone else with the email code.
- `StepUpAuthService.java` rules:
  - **Who must step up:** pending admins are refused. Admins use TOTP. Companies and influencers use an email code, but are skipped entirely while initial account setup is incomplete (l.56-86).
  - **Email code:** 6 digits, stored hashed in Redis for 10 minutes, sent to the user's last verified email.
  - **Brute-force limits:** 5 attempts per code, then a 15-minute cooldown; 3 cooldowns lead to a 24-hour lockout.
  - **Token:** a UUID stored in Redis for 10 minutes. The Redis key includes the user and the action type, so a token only works for the action it was issued for.
  - **Checking the token:** `validateAndConsumeToken` reads and deletes it in one step, so it works once. A missing or wrong token raises `AuthenticationTranslatableException`, which `GlobalDefaultExceptionHandler` turns into **401**. `validateTokenIfRequired` checks the token only when step-up is required.
- **Only three places enforce it**, each reading the `X-Step-Up-Token` header by hand:
  - `UserController.java:284` (PATCH) and `:315` (PUT), only when the email actually changes.
  - `FirebaseAuthProxyController.java:457` (password change).

**Billing endpoints** (subsystem [11]); none of them has step-up
- `subscription/SubscriptionPaidController.java`: company users only, registered only when `app.payments.enabled` is on, and hidden from the OpenAPI spec.

| endpoint | what it does | step-up today |
|---|---|---|
| `POST /subscription/upgrade` | creates a Stripe Checkout session (plan change, money) | **none** |
| `POST /subscription/downgrade` | plan change | **none** |
| `POST /subscription/downgrade/cancel` | cancels a scheduled downgrade | **none** |
| `POST /subscription/portal` | opens the Stripe customer portal, where cancelling and changing the payment method happen; the app has no endpoint of its own for either (`StripeWebhookHandler` handles `customer.subscription.deleted`) | **none** |
| `POST /subscription/trial/activate`, `POST /subscription/consent` | trial start, legal consent proof | none (proposed to stay open) |

- **Invoice data:** the invoice buyer's name, tax number and address come from `CompanyData` (`subscription/invoicing/InvoiceRetryService.java` uses `CompanyDataRepository`). It is managed by `registry/RegistryController.java`:
  - `POST /registry/confirm` works only once (409 if data already exists).
  - `DELETE /registry/company-data` resets it and `POST /registry/refresh` re-fetches it. **Neither has step-up.**

**Frontend** (subsystems [170], [174], [177])
- **Step-up plumbing:**
  - `core/step-up/step-up.service.ts` wraps check, request and verify.
  - `core/step-up/step-up-context.ts` holds the token in the request context (`STEP_UP_TOKEN`, set with `withStepUpToken()`).
  - `core/interceptors/step-up.interceptor.ts` turns that into the `X-Step-Up-Token` header.
  - `shared/components/step-up-dialog/step-up-dialog.component.ts` closes with the token, or with `null` both when the user cancels and when the backend says step-up isn't required.
- **Only caller today:** `feature/profile/email-change.component.ts`.
- **Action type model:** `api/model/step-up-action-type.ts` is generated from the OpenAPI spec, so a new backend value needs `openapi:gen`.
- **Billing client and call sites:**
  - `core/api-frozen/subscription.client.ts` (`SubscriptionPaidService`) is hand-written and cannot pass a request context.
  - `core/subscription/subscription.service.ts` (`SubscriptionWriteApi`) wraps it.
  - Callers: `feature/plan-billing/upgrade-confirm-dialog.component.ts:107`, `downgrade-confirm-dialog.component.ts:94`, and `plan-billing.component.ts:532` (cancel downgrade) and `:574` (portal).
- **Error handling:** `core/interceptors/error.interceptor.ts` treats a 401 as an expired session (silent refresh, then one retry). A 403 is passed back to the calling component.

## Proposed change
**One new action type, `BILLING_CHANGE`, enforced by an annotation instead of hand-written header checks.**

- **Backend:** add `@RequiresStepUp(StepUpActionType.BILLING_CHANGE)` and a `StepUpEnforcementAspect`. The aspect reads `X-Step-Up-Token`, takes the user id from `PermissionUtils.getUserId()`, and calls `validateTokenIfRequired`.
  - Put it on upgrade, downgrade, cancel-downgrade, portal, `DELETE /registry/company-data` and `POST /registry/refresh`.
- **Why an annotation:** the three current checks are copy-pasted into controllers. Six more copies, in two controllers, would be easy to forget on the next endpoint. An annotation can also be checked by a test that fails for any unguarded billing write.
- **Why one action type rather than one per endpoint:**
  - Lockout and cooldown are counted per action type. Per-endpoint types would give an attacker four separate 3-cooldown budgets.
  - The frontend and the generated enum stay small.
  - The cost: a token earned for one billing action could be spent on another billing action by the same user within 10 minutes. That is acceptable.
- **Kept the same as email change:** the 401 status and the skip while setup is incomplete. Changing to 403 would also change the contract that the email and password flows and `validation-edge-cases.feature` rely on; that can be a separate follow-up.
- **Left open on purpose:** `trial/activate` and `consent` (no money moves, and consent is part of onboarding) and `/registry/confirm` (first-time onboarding, blocked after that by its own 409).
- **Frontend:** before each billing write, call `stepUp.check(BILLING_CHANGE)`.
  - If required, open `StepUpDialogComponent` and send the token with `withStepUpToken`.
  - If not required, call the endpoint without a token.
  - Checking first avoids the dialog's ambiguous `null`, which today makes `email-change` silently stop for users whose setup is incomplete.

## Plan
1. **Backend enum and texts.** Add `BILLING_CHANGE` to `StepUpActionType.java`. Add any action-specific wording to `messages_en.properties` and `messages_pl.properties`, and to `templates/email/step-up-code.html` if the email names the action. Nothing uses the new value yet, so nothing changes in behaviour.
2. **Annotation and aspect.** Create `auth/stepup/RequiresStepUp.java` and `auth/stepup/StepUpEnforcementAspect.java`, modelled on `RecaptchaValidationAspect` / `RequiresRecaptcha` in [8]. The aspect must run after `@PreAuthorize`. Unit tests: missing token → 401; token issued for `EMAIL_CHANGE` → 401; valid token → proceeds, and the token cannot be used again; setup incomplete → proceeds; admin → TOTP path.
3. **Frontend first, still harmless.**
   - Run `openapi:gen` so the generated enum gets `BILLING_CHANGE`.
   - Add an optional `HttpContext` parameter to `initiateUpgrade`, `requestDowngrade`, `cancelDowngrade` and `createPortalSession` in `subscription.client.ts` and `subscription.service.ts`.
   - Add a small helper, e.g. `core/step-up/run-with-step-up.ts` (check → dialog → token or proceed; `null` means the user cancelled).
   - Use it in `upgrade-confirm-dialog.component.ts`, `downgrade-confirm-dialog.component.ts` and `plan-billing.component.ts` (cancel downgrade, portal), and in any caller of registry reset or refresh.
   - Update specs and the demo layer (`core/demo/demo.interceptor.ts`, `demo-fixtures.ts`, `scenario-registry.ts`) so demo billing flows still work.
   - The backend still ignores the header, so deploying this changes nothing yet.
4. **Turn on enforcement.** Annotate the four `SubscriptionPaidController` methods and the two `RegistryController` methods. Deploy after step 3 so real users never get a 401 they can't resolve.
5. **Guard test.** A reflection test finds every `@PostMapping`, `@PutMapping`, `@PatchMapping` and `@DeleteMapping` in `SubscriptionPaidController` and every change endpoint in `RegistryController`, and fails if one lacks `@RequiresStepUp`. The allowlist holds only `trial/activate`, `consent` and `registry/confirm`.
6. **End-to-end.** Add Cucumber scenarios in `e2e/steps/StepUpAuthSteps.java` plus a billing feature file: upgrade without a token → 401; request, verify, then upgrade → 200; replaying the token → 401. Add a frontend e2e/sandbox scenario for upgrade with step-up.
7. **Optional follow-up.** Move the three existing inline checks onto `@RequiresStepUp`, and decide separately whether a step-up failure should be 403 instead of 401.

## Risks and invariants
- **Payments flag off:** `SubscriptionPaidController` doesn't exist, so the new guard must not add any bean that depends on it. The aspect is generic, so this holds. *Test:* the context starts with `app.payments.enabled=false`.
- **Token works once, for the right user and action:**
  - Covered by the atomic read-and-delete and the Redis key.
  - The token is used up **before** the Stripe call. If Stripe then fails, the user has to step up again. That is acceptable, but the frontend must reopen the dialog on retry and never replay the old token.
  - *Test:* the replay scenario in step 6.
- **401 triggers a silent session refresh** in `error.interceptor.ts`. A billing call without a token costs one refresh and one retry, which fails again and shows the error; the user is not logged out. *Test:* a frontend spec that billing 401s show an error and do not clear the session.
- **Skip while setup is incomplete:** if a company whose setup isn't complete can reach the billing writes, they would skip step-up (checked in phase 2).
- **Admins:** `SubscriptionPaidController` is company-only, so the admin TOTP path isn't used there. `RegistryController` has no role check, so an admin there would be asked for TOTP. *Test:* the aspect unit test for the admin path.
- **Lockout is shared by all billing actions:** 3 cooldowns lock billing for 24 hours. Support needs to know how to clear the Redis key.
- **Release order:** frontend (step 3) must ship before enforcement (step 4), or the plan-billing page returns 401s.
- **Stripe portal link:** only the creation of the link is guarded. The returned URL is itself a bearer link, so it must never be logged or cached.

## Evidence
- FACT — the action types are `EMAIL_CHANGE` and `PASSWORD_CHANGE`; the challenge types are `EMAIL_CODE` and `TOTP` (backend `auth/stepup/StepUpActionType.java`, `StepUpChallengeType.java`).
- FACT — who must step up, the limits, the TTLs, the per-action key and the atomic read-and-delete (backend `StepUpAuthService.java` l.39-251, read in full).
- FACT — a missing or wrong token → `AuthenticationTranslatableException` → 401 (`GlobalDefaultExceptionHandler.java:97-98`).
- FACT — the only enforcement points are `UserController.java:284` and `:315`, and `FirebaseAuthProxyController.java:457` (grep across `src/main`).
- FACT — upgrade, downgrade, cancel-downgrade and portal have no step-up (`SubscriptionPaidController.java`, read in full).
- FACT — the invoice buyer data comes from `CompanyData`, and the registry reset and refresh endpoints have no step-up (`InvoiceRetryService.java:6-22,89`; `RegistryController.java:92-116`).
- FACT — the frontend has the token context, the interceptor and the dialog (dialog closes with `null` when step-up isn't required); `email-change` is the only caller; the subscription client has no context parameter (files read).
- FACT — the frontend enum is generated (`api/model/step-up-action-type.ts`, header "auto generated").
- FACT — the frontend's 401 handling is refresh and retry; 403 is passed to the caller (`error.interceptor.ts:17-35,84-114`).
- INFERENCE — cancelling and changing the payment method happen only in the Stripe portal, because there is no in-app endpoint for either and the webhook handles `customer.subscription.deleted`. Guarding `/portal` therefore covers them.
- HYPOTHESIS — billing writes can't be reached while setup is incomplete, so the skip doesn't weaken the guard.
- HYPOTHESIS — `trial/activate` moves no money.
- HYPOTHESIS — the step-up email doesn't name the action, so it reads fine for billing.
- HYPOTHESIS — a recaptcha aspect exists to copy.

## Corrections after verification

**Corrections:**

1. **Wrong: "billing writes can't be reached while setup is incomplete."**
   - `SubscriptionService.initiateUpgrade` (l.136-148), `requestDowngrade` (l.442+) and `activateTrial` (l.60-72) only check the payments flag and subscription state.
   - Grep finds no reference to `subscription`, `registry` or setup/account status in `EmailVerificationEnforcementFilter`, `BannedUserAuthorizationFilter` or `ConsentEnforcementFilter`.
   - As far as the code shows, a company whose setup is incomplete can reach the billing writes. Reusing the setup-incomplete skip would leave them unguarded.
   - **Change to step 2:**
     - For `BILLING_CHANGE`, never skip. In `StepUpAuthService.checkRequirement`, branch on the action: return `required=true` (email code, or TOTP for admins) whatever the setup state.
     - The aspect calls `validateAndConsumeToken` directly, not `validateTokenIfRequired`.
     - If the user has no `lastVerifiedEmail`, `requestCode` throws `InsufficientPermissionsException` with a new key, e.g. `error.stepup.billing_requires_verified_email` (403). Otherwise there is nowhere to send the code.
     - Add aspect unit tests: setup incomplete + `BILLING_CHANGE` → still 401 without a token; no verified email → 403 on `/step-up/request`.
   - The Risks item about the skip is replaced by this. In step 3, the frontend should treat `required=false` for `BILLING_CHANGE` as unexpected and not proceed.
2. **Can't be settled from the code: what the Stripe portal allows.** `StripeService.createPortalSession` (l.161-171) passes only the customer and return URL, so cancelling and changing the payment method are controlled by the default portal configuration in the Stripe dashboard. The claim that guarding `/portal` covers them holds only if that is where users do them. The payments owner needs to confirm the dashboard settings; otherwise add portal `flow_data` limits.
3. **Confirmed:** the step-up email is generic. The template uses only `email.step_up_code.*` keys and never names the action, so step 1 needs no template change.
4. **Confirmed:** there is a pattern to copy at backend `auth/config/RecaptchaValidationAspect.java` (a different package from the one the map suggested).
5. **Scope trim to step 3:** no frontend feature calls `resetCompanyData` or `refreshCompanyData` (grep over `frontend/src/app/feature`: none), so the registry needs no frontend work. Its guard is backend-only (steps 4-5).
6. **Still a hypothesis:** that `activateTrial` moves no money. No Stripe call appears in its first lines, but I didn't read the rest of the method. Leaving it unguarded stays in the allowlist for the owner to confirm.
