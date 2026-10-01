I've finished reading both repositories. Here is the analysis.

# Problem

The request is to require step-up authentication (a fresh email code, or TOTP for admins, just before a sensitive action) for every action that moves money or changes billing: plan changes, cancellations, payment-method changes and invoice data. Today the backend enforces step-up in exactly three places, and all of them are email or password changes. None of the billing endpoints check for a step-up token, and the frontend never asks for one on the billing page. Someone holding a stolen session cookie can upgrade a plan (which charges the saved card immediately), cancel, open the Stripe billing portal, or reset the company data used on invoices.

# Where it lives today

## Backend (`backend/src/main/java/com/sm/instagram/platform/`)

**How step-up works**
- **Action types** (`auth/stepup/StepUpActionType.java`): only `EMAIL_CHANGE` and `PASSWORD_CHANGE` exist.
- **Challenge types** (`auth/stepup/StepUpChallengeType.java`): `EMAIL_CODE` and `TOTP`.
- **Service** (`auth/stepup/StepUpAuthService.java`):
  - `checkRequirement` (l.56–86) decides by user type:
    - `PENDING_ADMIN` is refused with a 403.
    - `ADMIN` gets `TOTP`.
    - A `COMPANY` or `INFLUENCER` user whose `initialAccountSetupCompleted` is not true gets `required=false`, so step-up is skipped.
    - Everyone else gets `EMAIL_CODE`.
  - `requestCode` makes a 6-digit code, stores its SHA-256 hash in Redis for 10 minutes and emails it to `lastVerifiedEmail`.
  - `verifyCode` allows 5 tries per code, then a 15-minute cooldown. After 3 cooldowns the action is locked for 24 hours (HTTP 423).
  - On success it creates a token: a random UUID stored in Redis for 10 minutes under the key `step_up_token:<hash(uid)>:<ACTION>`.
  - `validateAndConsumeToken` reads and deletes the token in one step (`getAndDelete`), so it works once only and only for its own action. A bad token raises `AuthenticationTranslatableException`, which becomes **401** (`GlobalDefaultExceptionHandler.java:97`).
  - `validateTokenIfRequired` re-runs `checkRequirement`, so it skips the check whenever that returns `required=false`.
  - `verifyAdminTotp` has no attempt counter; it relies only on the controller's STRICT rate limit.
- **Controller** (`auth/stepup/StepUpAuthController.java`): `GET /step-up/check`, `POST /step-up/request`, `POST /step-up/verify`.
- **How the token travels:** in the `X-Step-Up-Token` header, which CORS already allows (`common/authorization/WebSecurityConfiguration.java:282`).

**Endpoints that enforce step-up (the only three)**

| Endpoint | Action | Where |
|---|---|---|
| `PATCH /users/{id}` (only when the email actually changes) | EMAIL_CHANGE | `user/UserController.java:282-286` |
| `PUT /users/{id}` (same rule) | EMAIL_CHANGE | `user/UserController.java:314-317` |
| `POST /auth/firebase/change-password` | PASSWORD_CHANGE | `auth/FirebaseAuthProxyController.java:457-458` |

**Billing endpoints with no step-up check**

Unless noted, all are in `subscription/SubscriptionPaidController.java`, which only exists when `app.payments.enabled` is on and requires the `COMPANY` authority.

| Endpoint | Effect |
|---|---|
| `POST /subscription/upgrade` | For a first purchase: creates a Stripe Checkout session. With an existing subscription: `updateSubscriptionPrice` using `ALWAYS_INVOICE`, which **charges the saved card immediately with no Stripe-hosted confirmation** (`SubscriptionService.java:164-182`, `StripeService.java:125-138`). |
| `POST /subscription/downgrade` | To FREE: `cancelSubscriptionAtPeriodEnd` (this is how users cancel), or an immediate trial cancellation (`SubscriptionService.java:447-500`). To a lower paid plan: creates a Stripe schedule. |
| `POST /subscription/downgrade/cancel` | Brings back the higher paid plan, so the user keeps paying (`SubscriptionService.java:512-550`). |
| `POST /subscription/portal` | Opens a Stripe Customer Portal session (`StripeService.java:161-171`). The portal is where users change payment methods, billing details and, depending on the Stripe dashboard setup, cancel. |
| `POST /subscription/trial/activate` | FREE → TRIAL_ENTERPRISE. This is a plan change and uses up the one trial. |
| `DELETE /registry/company-data` (`registry/RegistryController.java:97-103`) | Deletes `CompanyData`, which lets the user confirm a different NIP (Polish tax ID). Invoices are built from `CompanyData` (name, NIP, registered address) (`subscription/invoicing/InvoiceRetryService.java:78-93`). |

Endpoints that are **not** money or billing changes:
- `POST /subscription/consent` only records consent.
- `GET /subscription/config`, `/status` and `/invoices` are reads.
- `POST /registry/confirm` is a one-time first set-up; it throws if company data already exists (`RegistryLookupService.java:155`).
- `POST /registry/refresh` re-fetches data from the public registries; the user can't supply any values.

Two more things to know:
- `docs/StripeGateway/StepUpAuthentication.md` describes a `PAYMENT_METHOD_CHANGE` flow, Firestore storage and "no step-up for admins". The code does none of these: it uses Redis, gives admins TOTP, and has no payment action type. The doc was never implemented for billing.
- `PATCH /users/{id}` can set `user.nip` (`UserService.java:546`). That is a copied field; invoicing does not read it.

## Frontend (`frontend/src/app/`)

**Step-up plumbing**
- `core/step-up/step-up.service.ts` wraps the generated API: `check`, `request`, `verify`.
- `core/step-up/step-up-context.ts` defines `STEP_UP_TOKEN` and `withStepUpToken()`.
- `core/interceptors/step-up.interceptor.ts` adds the header whenever that context is set.
- `shared/components/step-up-dialog/step-up-dialog.component.ts` calls `/request` and closes with `null` if the response says `required:false`. On verify it closes with the token, and it maps 401, 429 and 423 to error messages.
- `api/model/step-up-action-type.ts` is generated from OpenAPI and only has the two current actions.
- The only feature that uses the dialog is `feature/profile/email-change.component.ts:107-132`.
- `feature/settings/security-settings.component.ts:91` calls `changePassword` **without** opening the dialog or sending a token.

**Billing**
- `core/api-frozen/subscription.client.ts` is a hand-written client, because the subscription controllers are `@Hidden` from OpenAPI. It takes no `HttpContext`.
- `core/subscription/subscription.service.ts` (`SubscriptionWriteApi`) wraps it.
- Call sites:
  - `feature/plan-billing/upgrade-confirm-dialog.component.ts:106-107` records consent, then upgrades.
  - `downgrade-confirm-dialog.component.ts:94` handles downgrades and trial cancellation.
  - `plan-billing.component.ts:389` activates the trial, `:532` cancels a pending downgrade, and `:574` opens the portal.
- `core/interceptors/error.interceptor.ts:40,113-136` treats **any** 401 from a logged-in user as a session problem: it refreshes the session and replays the request.

# Proposed change

1. **Add billing action types to `StepUpActionType`:** `PLAN_CHANGE` (upgrade, downgrade including cancel-to-FREE and trial cancel, cancelling a pending downgrade, trial activation), `BILLING_PORTAL_ACCESS` (the portal, which covers payment methods, billing details and cancellation inside Stripe) and `BILLING_DETAILS_CHANGE` (resetting company data).
   - Why one `PLAN_CHANGE` rather than a separate cancel type: cancelling is `/downgrade` with `targetPlan=FREE`. A separate type would force both sides to pick the action based on the request body, and it would add no security.
   - Why I named it `BILLING_PORTAL_ACCESS` rather than the doc's `PAYMENT_METHOD_CHANGE`: the portal does more than change cards.
2. **Billing actions never skip step-up.** Give the enum a property such as `skippableDuringInitialSetup()`: `true` for EMAIL_CHANGE and PASSWORD_CHANGE, `false` for billing. `checkRequirement` reads it. If step-up can't be done (for example `lastVerifiedEmail` is null), billing actions get a 403, not "not required". This has to live in `checkRequirement` itself, so `/check`, `/request` and the token check all agree. Otherwise `/request` could answer `required:false`, the dialog would close with `null`, and the protected call would fail with a 401.
3. **Check the token explicitly at the top of each guarded controller method.** Call `stepUpAuthService.validateAndConsumeToken(uid, ACTION, request.getHeader("X-Step-Up-Token"))` before any service or Stripe call. This matches the existing style in `UserController` and `FirebaseAuthProxyController`. A test that walks the registered endpoints (see Risks) stops a future billing endpoint from shipping without the check.
   - I'd use `validateAndConsumeToken` directly rather than `validateTokenIfRequired`, so billing always fails closed.
   - `SubscriptionPaidController` requires the `COMPANY` authority, so admins and influencers never reach it.
4. **Say which action the code is for in the email** (for example "to change your subscription plan"). Then a code requested by a stolen session is recognisable to the real owner. This means adding the action to `EmailService.sendStepUpCodeEmail` and adding `email.step_up_code.action.*` keys in both languages.
5. **Frontend: one small helper instead of copying the dialog code five times.** Add `StepUpGate.run(action, (token) => write$)` that opens `StepUpDialogComponent`, stops if it returns `null`, and otherwise makes the write with `withStepUpToken(token)`.
   - `SubscriptionPaidService` and `SubscriptionWriteApi` methods get an optional token or context argument, and so does the registry reset call.
   - In the upgrade dialog, step-up runs **before** `recordConsent`, so no consent is recorded for an upgrade that was abandoned.
6. **Don't let a failed step-up look like an expired session.** In `error.interceptor.ts`, skip the refresh when the request carried `STEP_UP_TOKEN`, or when the 401 body has the key `error.stepup.invalid_token`. Keeping the backend at 401 avoids breaking the existing email-change tests and the dialog's 401 handling.

Two things I'm not proposing:
- **Tying the token to request details** such as the target plan. The token already works once, only for one action, expires in 10 minutes and required the owner's email. Tying it to details would add complexity for little gain.
- **Adding step-up to `/registry/confirm` or `/registry/refresh`.** The first only runs once, and the second takes no user input. Resetting company data is the only way to change the invoice buyer.

# Plan

1. **Backend policy, without enforcement yet.** Add the three enum values and the no-skip property, update `checkRequirement`, add the action to the email text, and add the messages.
   - Files: `auth/stepup/StepUpActionType.java`, `auth/stepup/StepUpAuthService.java`, `support/common/EmailService.java`, `templates/email/step-up-code.html`, `messages_en.properties`, `messages_pl.properties`.
   - New test: `unit/.../StepUpAuthServiceUnitTest.java`. Nothing matching `*StepUp*` exists under `unit/` today.
2. **Regenerate the OpenAPI contract and the frontend model** so the frontend can ask for the new actions.
   - Files: `backend/docs/openapi/openapi.json`, `frontend/docs/openapi/openapi.json`, `frontend/src/app/api/model/step-up-action-type.ts`.
3. **Frontend client wiring.** Add the token/context argument and the new `StepUpGate` helper.
   - Files: `core/api-frozen/subscription.client.ts`, `core/subscription/subscription.service.ts`, `core/registry/registry.service.ts` (add a reset method if the UI gets one), `core/step-up/step-up-gate.ts` (new), `core/interceptors/error.interceptor.ts`.
4. **Frontend call sites.**
   - Files: `feature/plan-billing/upgrade-confirm-dialog.component.ts`, `downgrade-confirm-dialog.component.ts`, `plan-billing.component.ts` (trial, cancel pending downgrade, portal), `shared/components/step-up-dialog/*` (a title per action), `assets/i18n/en.json` and `pl.json`, `core/demo/demo-fixtures.ts` plus the sandbox fixtures (so demo mode handles `/step-up/*` for the new actions).
   - **Ship steps 1–4 before step 5.** Sending the header to a backend that doesn't check it yet does no harm, because CORS already allows it.
5. **Backend enforcement.** Add the token check to the five mutating methods in `SubscriptionPaidController` and to `resetCompanyData` in `RegistryController`.
   - Files: `subscription/SubscriptionPaidController.java`, `registry/RegistryController.java`.
6. **Tests and fixtures that call these endpoints directly.** They must now get a token first.
   - Files: `backend/src/test/java/.../e2e/steps/SubscriptionSteps.java` (l.71, 82, 89, reusing the `StepUpAuthSteps` "completes full step-up flow" step), `frontend/e2e-tests/_framework/api/subscription.api.ts` (l.86–96), `frontend/e2e-tests/integration/flows/subscription-lifecycle.spec.ts:155`.
7. **New tests** (listed in the next section), and update `docs/StripeGateway/StepUpAuthentication.md` to match what the code actually does.

# Risks and invariants

| Invariant or risk | Test that guards it |
|---|---|
| **I1:** No money or billing endpoint runs without a valid token for its own action. Stripe and the subscription service must not be touched when the token is missing or wrong. | New `SubscriptionPaidControllerUnitTest`, one parameterised case per endpoint: no header → 401 and `verifyNoInteractions(subscriptionService, stripeService)`; valid token → the service is called. Same pattern for `DELETE /registry/company-data` in `RegistryControllerUnitTest`. |
| **I2:** A future mutating billing endpoint can't ship without the check. | Endpoint-walk test: list every non-GET handler on `SubscriptionPaidController` and `RegistryController` from the registered request mappings, minus an explicit allowlist (`consent`, `lookup`, `confirm`, `refresh`), and assert each returns 401 without a token. |
| **I3:** Billing step-up never skips during initial setup, and fails closed when there is nowhere to send the code. | `StepUpAuthServiceUnitTest`: COMPANY with setup incomplete + PLAN_CHANGE → `required=true`; `lastVerifiedEmail == null` → 403; EMAIL_CHANGE still skips (the existing feature scenarios at l.88 and l.100 must stay green). |
| **I4:** Tokens only work for their own action and only once. | Unit test: a PLAN_CHANGE token is rejected for BILLING_PORTAL_ACCESS and EMAIL_CHANGE. New Cucumber file `features/step-up-billing.feature`: reusing a token → 401, and an EMAIL_CHANGE token on `/subscription/upgrade` → 401. |
| **I5:** With payments off, the paid endpoints still return 404, not 401. | Existing `payments_off/payments-off.feature` (backend) and `payments-off.spec.ts` (frontend) must stay unchanged and green. The check lives inside the conditionally created controller, so this should hold. |
| **R1:** Deploying enforcement before the frontend breaks billing in production. | Follow the plan order (steps 1–4 ship first). Frontend Jest tests on the plan-billing components: each action opens the dialog with the right action, cancelling makes no write call, and the token arrives in the request's context. |
| **R2:** A step-up 401 triggers a pointless session refresh and replays an already-used token. | `error.interceptor.spec.ts`: a 401 on a request carrying `STEP_UP_TOKEN` → no `refreshSession` call and the error reaches the caller. |
| **R3:** The token is used up even if the business rule then fails (for example "downgrade not allowed from this state"), so the user has to do step-up again. | Accepted. The frontend already hides actions that aren't allowed. A unit test documents that the token is used up before the service is called. |
| **R4:** Brute force spread across more actions. Each action gets 3×5 guesses per 24 hours, so the three new types add 45 guesses per day against 10⁶ possible codes. | Negligible. The existing "Five wrong codes triggers cooldown" scenario, repeated for PLAN_CHANGE. |
| **R5:** Upgrade dialog ordering: consent must not be recorded when step-up is cancelled. | `upgrade-confirm-dialog.component.spec.ts`: dialog returns `null` → `recordConsent` and `initiateUpgrade` are never called. |
| **R6:** Nothing guards what happens inside the Stripe portal after it opens. Also, `handleSubscriptionUpdated` only reacts to price changes, so a cancel-at-period-end set in the portal may not show up in the database. | Out of scope. Worth a follow-up: an integration test on the `customer.subscription.updated` webhook with `cancel_at_period_end=true`. |
| **End-to-end.** | Frontend Playwright `integration/auth/step-up-billing.spec.ts` reads the code from GreenMail (a local test mail server) via `_framework/test-email.ts`: downgrade → dialog → code → pending-downgrade banner appears; wrong code → error, and status unchanged. |

**Problems found along the way (existing, not caused by this change):**
- The password-change screen never sends a step-up token, but the backend demands one from users who have finished setup. Password change is probably broken for them today. I haven't run it to confirm.
- Admin TOTP step-up has no attempt limit.
- The design doc no longer matches the code.

# Evidence

**Step-up mechanism**
- FACT: Only two action types exist, EMAIL_CHANGE and PASSWORD_CHANGE — `StepUpActionType.java:3-6`, and the same in `frontend/src/app/api/model/step-up-action-type.ts:13-16`.
- FACT: The challenge types are EMAIL_CODE and TOTP; ADMIN gets TOTP, PENDING_ADMIN gets a 403, setup-incomplete COMPANY/INFLUENCER users skip — `StepUpAuthService.java:56-86`.
- FACT: Codes are a 6-digit SHA-256 hash in Redis for 10 minutes, with 5 tries, a 15-minute cooldown and a 24-hour lockout after 3 rounds — `StepUpAuthService.java:39-52, 88-178`.
- FACT: Tokens are UUIDs kept for 10 minutes, keyed per user and action, and read-and-deleted in one step — `StepUpAuthService.java:202-233, 249-251`.
- FACT: `validateTokenIfRequired` skips whenever the check says not required — `StepUpAuthService.java:194-200`.
- FACT: Admin TOTP verification has no attempt counter — `StepUpAuthService.java:180-188`, `TwoFactorAuthService.java:160-180`.
- FACT: An invalid token becomes a 401 — `GlobalDefaultExceptionHandler.java:97-98`.
- FACT: Step-up is enforced only for email change (PATCH and PUT) and password change — `UserController.java:282-286, 314-317`, `FirebaseAuthProxyController.java:457-458`, and a grep for `validateTokenIfRequired|validateAndConsumeToken` across `backend/src`.
- FACT: CORS already allows the `X-Step-Up-Token` header — `WebSecurityConfiguration.java:282`.

**Billing endpoints**
- FACT: The paid subscription endpoints check no token, and the controller only exists when payments are enabled — `SubscriptionPaidController.java:47, 60-107`.
- FACT: An upgrade on an existing subscription charges immediately via `ALWAYS_INVOICE` — `SubscriptionService.java:164-182`, `StripeService.java:125-138`.
- FACT: Downgrading to FREE cancels at period end, or cancels a trial immediately — `SubscriptionService.java:447-487`.
- FACT: The portal session is created with only the customer and return URL — `StripeService.java:161-171`.
- INFERENCE: The Stripe portal lets users change payment methods, billing details and possibly cancel. This depends on Stripe dashboard settings, not on code — backed by the frontend comment at `plan-billing.component.ts:557-558`.
- FACT: Invoice buyer data comes from `CompanyData` — `InvoiceRetryService.java:78-93`.
- FACT: Company data can only be changed by reset followed by confirm; confirm throws if data already exists — `RegistryController.java:97-103`, `RegistryLookupService.java:155, 247-264`.
- FACT: Refresh only uses registry data — `RegistryLookupService.java:275-328`.
- FACT: `PATCH /users/{id}` can set `user.nip` — `UserService.java:546`.
- HYPOTHESIS: Account owners, not only admins, can patch `nip`. I did not read the permission check behind `validateUserUpdatePermission`.

**Frontend**
- FACT: The frontend step-up flow (context token, interceptor, dialog closing with `null` when not required) — `step-up.interceptor.ts:13-25`, `step-up-context.ts:11-19`, `step-up-dialog.component.ts:101-115`.
- FACT: Only the email-change feature uses the dialog — grep across `feature/`, `email-change.component.ts:107-132`.
- FACT: The password-change screen sends no token — `security-settings.component.ts:91`.
- HYPOTHESIS: Password change therefore fails today for users who have finished setup. Not run.
- FACT: The billing client has no context argument, and the call sites don't use step-up — `subscription.client.ts:68-108`, `upgrade-confirm-dialog.component.ts:106-107`, `downgrade-confirm-dialog.component.ts:94`, `plan-billing.component.ts:389, 532, 574`.
- FACT: Any logged-in 401 triggers a session refresh and a replay — `error.interceptor.ts:40, 82-136`.
- INFERENCE: The 401 response body includes the message key `error.stepup.invalid_token`. The body format wasn't read; check it before relying on it in the interceptor.

**Other code facts**
- FACT: The subscription controllers are `@Hidden` from OpenAPI, which is why the frontend client is hand-written — `SubscriptionController.java:31`, `SubscriptionPaidController.java:42`, `subscription.client.ts:1-13`.
- FACT: The email template has no line saying which action the code is for — `step-up-code.html:54`, `messages_en.properties:624-629`.
- FACT: `handleSubscriptionUpdated` only reacts to price changes — `SubscriptionService.java:380-435`.
- INFERENCE: A cancel-at-period-end set inside the Stripe portal may not reach the database.
- INFERENCE: `initialAccountSetupCompleted` can be false while `lastVerifiedEmail` is null, so billing needs the 403 rule. The field defaults to false (`User.java:162`); I did not read where `lastVerifiedEmail` gets set.

**Tests and docs**
- FACT: Existing tests call the billing endpoints directly without a token and will need one — `SubscriptionSteps.java:71, 82, 89`, `frontend/e2e-tests/_framework/api/subscription.api.ts:86-96`, `subscription-lifecycle.spec.ts:155`.
- FACT: The payments-off tests expect the paid endpoints to be absent — `backend/src/test/resources/features/payments_off/payments-off.feature:33-39`, `frontend/e2e-tests/integration/flows/payments-off.spec.ts:107-109`.
- FACT: There is no unit test for `StepUpAuthService` or `SubscriptionPaidController` — glob `backend/src/test/java/**/*{StepUp,Subscription,Registry}*.java`.
- FACT: The design doc (Firestore storage, no admin step-up, `PAYMENT_METHOD_CHANGE`) doesn't match the code — `docs/StripeGateway/StepUpAuthentication.md:11, 15, 63-127, 198-212` compared with the code above.
