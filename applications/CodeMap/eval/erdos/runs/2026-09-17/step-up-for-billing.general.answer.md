# Step-up authentication for billing actions: current state and proposed change

## Problem

Today step-up authentication only protects **email change** and **password change**. None of the billing endpoints ask for it. That covers plan upgrades (which charge the card), downgrades and cancellations, undoing a scheduled downgrade (billing resumes), opening the Stripe Customer Portal (where the card is changed and the subscription can be cancelled), and editing the company tax data (NIP) printed on invoices. Anyone holding a stolen session cookie can do all of it. The goal is that every one of these actions needs a fresh, single-use step-up token.

## Where it lives today

**Backend step-up module:** `backend/src/main/java/com/sm/instagram/platform/auth/stepup/`
- `StepUpActionType.java` has only `EMAIL_CHANGE` and `PASSWORD_CHANGE`.
- `StepUpChallengeType.java` has `EMAIL_CODE` and `TOTP`.
- `StepUpAuthService.java`:
  - `checkRequirement` (:56): `PENDING_ADMIN` is refused. `ADMIN` gets TOTP. For COMPANY and INFLUENCER users who haven't finished account setup, it returns `required=false`. Everyone else gets `EMAIL_CODE`.
  - `requestCode` (:88): makes a 6-digit code, stores its SHA-256 hash in Redis for 10 minutes and sends it to `lastVerifiedEmail`.
  - `verifyCode` (:121): 5 attempts per code, then a 15-minute cooldown; 3 cooldowns lead to a 24-hour lockout (HTTP 423).
  - `generateAndStoreToken` (:220): a UUID stored in Redis for 10 minutes. The key is per user **and per action**.
  - `validateAndConsumeToken` (:202): reads and deletes the token in one step (`getAndDelete`), so it can be used once. A bad or missing token gives 401 (`error.stepup.invalid_token`).
  - `validateTokenIfRequired` (:194): skips the check when `checkRequirement` says it isn't needed.
- `StepUpAuthController.java`: `GET /step-up/check`, `POST /step-up/request`, `POST /step-up/verify`, with `RateLimitProfile.STRICT`.

**Where the token is enforced today** (all read the `X-Step-Up-Token` header and call `validateTokenIfRequired`):
- `user/UserController.java:282-286` (PATCH) and `:314-317` (PUT): only when the email actually changes.
- `auth/FirebaseAuthProxyController.java:457-458`: `POST /auth/firebase/change-password`.

**Billing endpoints with no step-up:**
- `subscription/SubscriptionPaidController.java` (COMPANY only, only exists when `app.payments.enabled` is on):
  - `POST /subscription/upgrade`: creates a Stripe Checkout session, or changes the price in place with `ALWAYS_INVOICE`, which charges immediately (`SubscriptionService.java:164-170`, `StripeService.java:125-137`).
  - `POST /subscription/downgrade`: moving to FREE is a cancellation (`cancelSubscriptionAtPeriodEnd`). From a trial it cancels immediately (`SubscriptionService.java:447-486`).
  - `POST /subscription/downgrade/cancel`: brings billing back (`:512-531`).
  - `POST /subscription/portal`: returns a Stripe Customer Portal URL, where the card is changed and the plan can be cancelled (`StripeService.java:161-171`).
  - `POST /subscription/trial/activate`: plan change to an Enterprise trial.
  - `POST /subscription/consent`: only records consent, so I leave it unprotected.
- `registry/RegistryController.java`: `POST /registry/confirm`, `DELETE /registry/company-data`, `POST /registry/refresh`. This stored company data is the invoice buyer (name, NIP, address) used for Fakturownia invoices (`subscription/invoicing/InvoiceRetryService.java:78-89`).
- `subscription/TestSubscriptionController.java` only exists under the `(e2e | dev) & !prod & !test` profiles, so it's out of scope.

**Frontend:**
- `core/step-up/step-up.service.ts` wraps check, request and verify.
- `core/step-up/step-up-context.ts` defines the `STEP_UP_TOKEN` context and `withStepUpToken`.
- `core/interceptors/step-up.interceptor.ts` adds the `X-Step-Up-Token` header.
- `shared/components/step-up-dialog/step-up-dialog.component.ts` closes with the token, or with `null` both on cancel and when step-up isn't required.
- `feature/profile/email-change.component.ts` is the only place that uses it (alongside `core/user/user.service.ts:40`).
- `api/model/step-up-action-type.ts` is generated from `docs/openapi/openapi.json` by `npm run openapi:gen`.
- The billing calls go through a hand-written client, `core/api-frozen/subscription.client.ts`, because the backend hides these controllers from OpenAPI (`@Hidden`). Wrapper: `core/subscription/subscription.service.ts`. Callers: `feature/plan-billing/upgrade-confirm-dialog.component.ts:106-107`, `downgrade-confirm-dialog.component.ts:94`, and `plan-billing.component.ts:389` (trial), `:532` (undo downgrade), `:574` (portal).
- **Pitfall:** `core/interceptors/error.interceptor.ts:113-135` answers any 401 from a signed-in user by refreshing the session and retrying. Its `catchError` sits after `switchMap`, so a 401 on the retry counts as a "hard auth failure": it clears the session and sends the user to `/auth/sign-in`. A step-up 401 (missing or already-used token) would therefore **sign the user out**.

## Proposed change

1. **New action types, one per kind of risk:** `PLAN_CHANGE`, `SUBSCRIPTION_CANCEL`, `PAYMENT_METHOD_CHANGE`, `BILLING_DETAILS_CHANGE`. Tokens are already tied to one action in Redis, so a token for "downgrade to Business" can't be spent on "cancel". Each action type also gets its own audit entry and its own lockout.
2. **Billing fails closed.** Add `StepUpAuthService.requireAndConsumeToken(uid, action, token)`, backed by a billing-aware `checkRequirement`:
   - For `PLAN_CHANGE`, `SUBSCRIPTION_CANCEL` and `PAYMENT_METHOD_CHANGE`, the "setup incomplete, skip" shortcut doesn't apply. If there's no `lastVerifiedEmail`, return 409 `error.stepup.verified_email_required` instead of letting the action through.
   - `BILLING_DETAILS_CHANGE` keeps the existing skip, because `/registry/confirm` is part of onboarding and can activate the account. The data only feeds invoices once there's a paid subscription.
   - Email and password changes don't change at all.
3. **Enforce in the controllers**, the same way `UserController` does (read the header, call the service before any business logic):
   - upgrade and trial/activate → `PLAN_CHANGE`
   - downgrade → `SUBSCRIPTION_CANCEL` when the target is FREE, otherwise `PLAN_CHANGE`
   - downgrade/cancel → `PLAN_CHANGE`
   - portal → `PAYMENT_METHOD_CHANGE`
   - registry confirm, reset and refresh → `BILLING_DETAILS_CHANGE`

   The token is used up before Stripe is called, so a Stripe error means doing step-up again. That is the safe choice and matches today's pattern.
4. **Portal:** requiring step-up to create the portal session is the only control we have. What the user can do once inside the portal is set in Stripe, not in this code (see the Evidence section).
5. **Frontend:**
   - Add `withStepUpToken` options to every write method in `subscription.client.ts` / `SubscriptionWriteApi`.
   - Open `StepUpDialogComponent` before the write: in the upgrade dialog before consent, so no orphan consent row is left behind; in the downgrade dialog, `activateTrial`, `cancelPendingDowngrade`, `openCustomerPortal`, and the registry flows.
   - If the dialog returns `null` for a billing action, stop.
   - In `error.interceptor.ts`, don't refresh or sign the user out when the 401 body's `messageKey` starts with `error.stepup.`. The backend sends `messageKey` (`GlobalDefaultExceptionHandler.java:88,97-98`).

## Plan

1. **Stop step-up errors from signing users out (do this first; it's an existing bug).** In `frontend/src/app/core/interceptors/error.interceptor.ts`, skip the refresh when `err.error?.messageKey?.startsWith('error.stepup.')`, and add tests to `error.interceptor.spec.ts`.
2. **Backend enum and service.**
   - `StepUpActionType.java`: add the four values.
   - `StepUpAuthService.java`: add `requireAndConsumeToken`, make the setup-incomplete skip depend on the action, and return 409 when there's no verified email.
   - `StepUpAuthController.java` needs no change: `/request` already returns `required:false` or the challenge type.
   - Add the new message keys to `messages_en.properties` and `messages_pl.properties`.
3. **Enforce on the endpoints.**
   - `SubscriptionPaidController.java`: inject `StepUpAuthService` and `HttpServletRequest` and guard the five writes.
   - `RegistryController.java`: guard confirm, reset and refresh.
4. **API contract.**
   - Update `backend/docs/openapi/openapi.json`, then copy it to `frontend/docs/openapi/openapi.json`, run `npm run openapi:gen`, and check that `frontend/src/app/api/model/step-up-action-type.ts` has the new values.
   - Add the new registry header to the OpenAPI annotations.
5. **Frontend clients.**
   - Add an optional `stepUpToken` to the write methods in `core/api-frozen/subscription.client.ts` and `core/subscription/subscription.service.ts`.
   - Registry calls use the generated `registry.api.ts`, which already accepts `options.context`.
6. **Frontend flows.**
   - `upgrade-confirm-dialog.component.ts`, `downgrade-confirm-dialog.component.ts` and `plan-billing.component.ts` (trial, undo downgrade, portal).
   - The registry and company-data components that call confirm, reset and refresh.
   - i18n `en.json` and `pl.json` (dialog text for each action and the new 409 key).
7. **Demo and sandbox.** `core/demo/demo-fixtures.ts` matches step-up routes by path regex, so it probably works as is. Make sure the demo plan-billing tour still runs, and update `sandbox/fixtures/step-up-dialog.fixture.ts` if it lists actions.
8. **Docs.** Update `backend/docs/StripeGateway/StepUpAuthentication.md` (scope and §6) to describe what is actually built.

## Risks and invariants, with the tests that guard them

| Invariant | Guarding test (new unless noted) |
|---|---|
| Every write mapping in `SubscriptionPaidController` and every registry write returns 401 without a token and runs no business logic | `unit/controller/SubscriptionPaidControllerStepUpUnitTest` (MockMvc, `verifyNoInteractions(subscriptionService, stripeService)`). Also a reflection guard test that lists every `@PostMapping`/`@DeleteMapping` in both controllers and fails if one isn't covered, so a future endpoint can't skip step-up. |
| A token is tied to one action and one use | New `unit/service/StepUpAuthServiceUnitTest`: a `PLAN_CHANGE` token is refused for `SUBSCRIPTION_CANCEL`; a second use gets 401. No unit test for `StepUpAuthService` exists today. |
| Billing actions don't take the setup-incomplete shortcut; `BILLING_DETAILS_CHANGE` does | Same unit test, plus BDD scenarios in `backend/src/test/resources/features/step-up-auth.feature` (next to the existing "incomplete setup" ones). |
| Downgrade to FREE needs `SUBSCRIPTION_CANCEL`; downgrade to BUSINESS needs `PLAN_CHANGE` | Controller unit test; `SubscriptionSteps` / `RunSubscriptionIT` scenario |
| The Stripe portal URL is never returned without a token | Controller unit test (`stripeService.createPortalSession` never called) |
| Email and password step-up behave as before | Existing: `UserControllerUnitTest`, `FirebaseAuthProxyControllerUnitTest`, `step-up-auth.feature`, `frontend/e2e-tests/integration/flows/step-up-email-required.spec.ts` |
| A step-up 401 doesn't sign the user out; a real session 401 still does | `error.interceptor.spec.ts`: two new cases |
| The frontend never sends a billing write without a token, and cancelling the dialog sends nothing | Specs for `upgrade-confirm-dialog`, `downgrade-confirm-dialog` and `plan-billing.component` |
| The existing subscription and registry suites still pass once tokens are needed | Update `SubscriptionSteps.java` and `RegistrySteps.java` to fetch a token (reuse the code-capture hook in `StepUpHooks.java`); `RegistryControllerUnitTest` |

**Other risks:**
- **The per-action rate limit and cooldown now also apply to billing.** A user locked out of `PLAN_CHANGE` for 24 hours can't upgrade.
- **Onboarding friction** if `/registry/confirm` were locked down harder than planned. The design avoids this.
- **The portal session stays valid after step-up.** The step-up only gates creating it.

## Evidence

1. FACT: The action types are only `EMAIL_CHANGE` and `PASSWORD_CHANGE`; the challenge types are `EMAIL_CODE` and `TOTP` (`StepUpActionType.java`, `StepUpChallengeType.java`).
2. FACT: Tokens are UUIDs kept 10 minutes in Redis under a per-user, per-action key, and are used up with `getAndDelete` (`StepUpAuthService.java:220-223, 207-210, 249-251`).
3. FACT: Setup-incomplete COMPANY and INFLUENCER users get `required=false`; ADMIN gets TOTP (`StepUpAuthService.java:62-85`).
4. FACT: Token checks happen only at `UserController.java:285,316` and `FirebaseAuthProxyController.java:458`. A search of `backend/src/main` found no other callers.
5. FACT: None of the subscription or registry write endpoints check a step-up token (`SubscriptionPaidController.java`, `RegistryController.java`).
6. FACT: Upgrading an existing subscription charges immediately (`ALWAYS_INVOICE`); downgrading to FREE cancels at period end; cancelling a trial is immediate (`StripeService.java:125-147`, `SubscriptionService.java:447-486`).
7. FACT: Company data (NIP, name, address) is the Fakturownia invoice buyer (`InvoiceRetryService.java:78-89`, `FakturowniaAdapter.java:73-79`).
8. FACT: A bad or missing token gives 401 with `messageKey`, and the frontend error interceptor signs the user out when the retry also returns 401 (`GlobalDefaultExceptionHandler.java:88,97-98`; `error.interceptor.ts:114-135`).
9. FACT: The step-up interceptor runs first in the chain, before the error interceptor (`app.config.ts:59-65`), so the retry resends the header that was already used.
10. FACT: The billing frontend uses a hand-written client with no step-up context (`subscription.client.ts`). The generated action enum comes from `docs/openapi/openapi.json` (`package.json:63`).
11. FACT: The code in `requestCode` goes to `lastVerifiedEmail` (`StepUpAuthService.java:109`).
12. FACT: The demo fixtures match `/step-up/*` by path only (`demo-fixtures.ts:1463-1484`).
13. FACT: No unit test for `StepUpAuthService` or `SubscriptionPaidController` exists (glob of `backend/src/test`).
14. INFERENCE: The portal lets users change the card and cancel. The frontend comment at `plan-billing.component.ts:557-562` says "payment-method updates… plan management", but the Stripe portal configuration lives in the Stripe dashboard, not in the code.
15. HYPOTHESIS: A setup-incomplete COMPANY user can reach `/subscription/*`. Authorities come from the JWT role (`JwtAuthenticationFilter.java:238`), but `EmailVerificationEnforcementFilter` might block them. The fail-closed rule in step 2 matters only if they can.
16. INFERENCE: `/registry/confirm` is part of onboarding (its OpenAPI description says it "may auto-activate the account"), so a hard step-up there would break sign-up for users with no verified email.
17. HYPOTHESIS: `backend/docs/openapi/openapi.json` is regenerated by a build step rather than edited by hand.

## Corrections after verification

I checked the claims the plan depends on.

- **#15 (a COMPANY user who hasn't finished setup can reach `/subscription/*`): mostly confirmed.**
  - `EmailVerificationEnforcementFilter.java:51-56,76` only blocks POST to `/partnership-opportunity` and `/applied-opportunity`. Nothing there blocks subscription or registry calls.
  - Authorities come from the JWT role claim (`JwtAuthenticationFilter.java:238`), and `SubscriptionPaidController` only asks for `COMPANY`.
  - I found no other gate on setup or account status in this pass, but I didn't read every filter. So the fail-closed rule for billing is needed: with the current `validateTokenIfRequired`, such a user would skip step-up on billing.
- **#16 (`/registry/confirm` is part of onboarding): confirmed.**
  - `frontend/src/app/core/registry/registry.service.ts:8-15` says `confirm` "auto-activates the account when the email is already verified". The only caller is `feature/company/company-setup.component.ts:134`.
  - Confirm only works once (`RegistryLookupService.java:155-157`), so changing company data later means reset, then confirm.
- **#17 (`openapi.json` is generated): confirmed, but step 4 was wrong.** `backend/docs/openapi/REGENERATE.md` says the file is written by `OpenApiSpecGeneratorTest` and must never be edited by hand. `CODEOWNERS:26` says the same.
- **#14 (what the Stripe portal allows): still unconfirmed.** The code passes only the customer and return URL (`StripeService.java:165-168`), so what users can do inside the portal is set in the Stripe dashboard. Someone should check the dashboard before we count cancellation inside the portal as covered.

**Corrections:**
1. **Plan step 4:** don't edit `backend/docs/openapi/openapi.json` by hand. Regenerate it with `./mvnw verify -Pintegration -DskipPmd=true -Dfailsafe.includes='**/OpenApiSpecGeneratorTest.java'` (needs Docker and JDK 21), then run `npm run openapi:cycle:fast` in the frontend, which copies the spec across and regenerates the client. Commit as three separate pieces: backend change, spec diff, regenerated client. The subscription controllers stay `@Hidden`, so only the `StepUpActionType` enum and the registry endpoints change in the contract.
2. **Plan step 6:** the frontend never calls `resetCompanyData` or `refreshCompanyData`; the wrapper `core/registry/registry.service.ts` only exposes `lookup`, `confirm` and `companyData`. So the only frontend work for registry is `CompanyRegistryService.confirm` (accept a token) and `feature/company/company-setup.component.ts` (open the dialog only when `/step-up/check` says it's required). Reset and refresh only need backend enforcement and backend tests.
3. **Evidence #15** becomes: "FACT (partial): `EmailVerificationEnforcementFilter` doesn't block billing or registry endpoints; no other setup gate found." **Evidence #16 and #17** become FACT, with the sources above.
