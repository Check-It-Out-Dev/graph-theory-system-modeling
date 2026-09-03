# Q1 gold FULL — digest (auto answers shown; HAND prefixes applied where present)

## BE01 [EXECUTED] — What does PaymentsDisabledBootGuard do and when does it stop the app from booting?
rows=2 fp=9b61514ef5bfab34
AUTO: Located: PaymentsDisabledBootGuard.java (Rule, sub-11); PaymentsDisabledBootGuardUnitTest.java (Rule, sub-11). Paths in gold rows.
```
PaymentsDisabledBootGuard.java | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/config/PaymentsDisabledBootGuard.java
PaymentsDisabledBootGuardUnitTest.java | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/subscription/PaymentsDisabledBootGuardUnitTest.java
```

## BE02 [EXECUTED] — Which cron jobs does the backend run and on what schedules?
rows=9 fp=d2e5125064fb24b0
AUTO: Located: AnonymousConsentCleanupCronJob.java (Actor, sub-11); ConsentEnforcementCronJob.java (Actor, sub-11); DeferredDeletionCronJob.java (Actor, sub-5); EmailCronJob.java (Actor, sub-12); InvoiceRetryCronJob.java (Actor, sub-11); NoConsentAccountCleanupCronJob.java (Actor, sub-11) (+3 more). Paths in gold rows. Schedules live in each class's @Scheduled/@SchedulerLock annotations — content, not structure.
```
AnonymousConsentCleanupCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/AnonymousConsentCleanupCronJob.java
ConsentEnforcementCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/ConsentEnforcementCronJob.java
DeferredDeletionCronJob.java | Actor | 5 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/cron/DeferredDeletionCronJob.java
EmailCronJob.java | Actor | 12 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/notification/email/EmailCronJob.java
InvoiceRetryCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/cron/InvoiceRetryCronJob.java
NoConsentAccountCleanupCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/NoConsentAccountCleanupCronJob.java
SubscriptionPeriodProcessorCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/cron/SubscriptionPeriodProcessorCronJob.java
TermsGraceProcessorCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/cron/TermsGraceProcessorCronJob.java
TrialExpiryNotifierCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/cron/TrialExpiryNotifierCronJob.java
```

## BE03 [EXECUTED] — What Spring profiles exist (dev-lite, e2e, no-redis, prod-standalone) and when do I use each?
rows=13 fp=173f7d65bbf3cd1a
AUTO: Located: application-actuator.yml (Context, sub-3); application-dev-lite.yml (Context, sub-3); application-dev.yml (Context, sub-3); application-e2e.yml (Context, sub-3); application-integration.yml (Context, sub-3); application-monitoring.yml (Context, sub-3) (+7 more). Paths in gold rows.
```
application-actuator.yml | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/resources/application-actuator.yml
application-dev-lite.yml | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/resources/application-dev-lite.yml
application-dev.yml | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/resources/application-dev.yml
application-e2e.yml | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/resources/application-e2e.yml
application-integration.yml | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/resources/application-integration.yml
application-monitoring.yml | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/resources/application-monitoring.yml
application-no-redis.yml | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/resources/application-no-redis.yml
application-prod-standalone.yml | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/resources/application-prod-standalone.yml
application-prod.yml | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/resources/application-prod.yml
application-ssl.yml | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/resources/application-ssl.yml
```

## BE04 [EXECUTED] — What subscription plans and statuses does the system model?
rows=4 fp=0931e894e32b4381
AUTO: Located: SubscriptionPlan.java (Resource, sub-11); SubscriptionPlanRepository.java (Resource, sub-11); SubscriptionStatus.java (Resource, sub-11); SubscriptionStatusDtoOut.java (Resource, sub-11). Paths in gold rows.
```
SubscriptionPlan.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/entity/SubscriptionPlan.java
SubscriptionPlanRepository.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/repository/SubscriptionPlanRepository.java
SubscriptionStatus.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/entity/SubscriptionStatus.java
SubscriptionStatusDtoOut.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/dto/SubscriptionStatusDtoOut.java
```

## BE05 [EXECUTED] — How does the backend integrate with Fakturownia for invoicing?
rows=9 fp=be0f8730c688bf91
AUTO: Located: FakturowniaAdapter.java (Resource, sub-11); FakturowniaAdapterUnitTest.java (Rule, sub-11); FakturowniaAdapter_IntegrationTest.java (Rule, sub-11); FakturowniaConfig.java (Context, sub-11); FakturowniaCreateRequest.java (Resource, sub-11); FakturowniaInvoiceResponse.java (Resource, sub-11) (+3 more). Paths in gold rows.
```
FakturowniaAdapter.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/invoicing/FakturowniaAdapter.java
FakturowniaAdapterUnitTest.java | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/subscription/FakturowniaAdapterUnitTest.java
FakturowniaAdapter_IntegrationTest.java | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/integration/service/subscription/FakturowniaAdapter_IntegrationTest.java
FakturowniaConfig.java | Context | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/invoicing/FakturowniaConfig.java
FakturowniaCreateRequest.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/invoicing/dto/FakturowniaCreateRequest.java
FakturowniaInvoiceResponse.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/invoicing/dto/FakturowniaInvoiceResponse.java
FakturowniaProperties.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/invoicing/FakturowniaProperties.java
InvoicingPort.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/invoicing/InvoicingPort.java
fakturownia-sim.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/demo/sims/fakturownia-sim.component.ts
```

## BE06 [EXECUTED] — What does ConsentEnforcementFilter block and for whom?
rows=2 fp=ba118e301c0908b9
AUTO: Located: ConsentEnforcementFilter.java (Rule, sub-11); ConsentEnforcementFilterUnitTest.java (Rule, sub-11). Paths in gold rows.
```
ConsentEnforcementFilter.java | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/authorization/ConsentEnforcementFilter.java
ConsentEnforcementFilterUnitTest.java | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/ConsentEnforcementFilterUnitTest.java
```

## BE07 [EXECUTED] — Where are API rate limits enforced and configured?
rows=31 fp=bd17947c2ff1ffff
AUTO: Located: DynamicRateLimit.java (Rule, sub-3); FirebaseRateLimitConfiguration.java (Context, sub-3); GdprCompliantRateLimiterService.java (Process, sub-3); GdprRateLimitProperties.java (Resource, sub-3); GdprRateLimiterUnitTest.java (Rule, sub-0); InMemoryRateLimiterService.java (Process, sub-3) (+25 more). Paths in gold rows.
```
DynamicRateLimit.java | Rule | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/ratelimit/DynamicRateLimit.java
FirebaseRateLimitConfiguration.java | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/ratelimit/FirebaseRateLimitConfiguration.java
GdprCompliantRateLimiterService.java | Process | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/ratelimit/GdprCompliantRateLimiterService.java
GdprRateLimitProperties.java | Resource | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/ratelimit/GdprRateLimitProperties.java
GdprRateLimiterUnitTest.java | Rule | 0 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/GdprRateLimiterUnitTest.java
InMemoryRateLimiterService.java | Process | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/ratelimit/InMemoryRateLimiterService.java
InMemoryRateLimiterServiceUnitTest.java | Rule | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/InMemoryRateLimiterServiceUnitTest.java
InMemoryRateLimiterUnitTest.java | Rule | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/common/ratelimit/InMemoryRateLimiterUnitTest.java
InMemoryStorageRateLimitService.java | Actor | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/storage/service/InMemoryStorageRateLimitService.java
InMemoryStorageRateLimitServiceUnitTest.java | Rule | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/InMemoryStorageRateLimitServiceUnitTest.java
```

## BE08 [EXECUTED] — What is step-up authentication and which endpoints require it?
rows=11 fp=05f3bfe34fb8092f
AUTO: Located: RunStepUpAuthIT.java (Rule, sub-9); StepUpActionType.java (Resource, sub-9); StepUpAuthController.java (Actor, sub-9); StepUpAuthService.java (Process, sub-9); StepUpAuthSteps.java (Resource, sub-9); StepUpChallengeType.java (Resource, sub-6) (+5 more). Paths in gold rows.
```
RunStepUpAuthIT.java | Rule | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/e2e/RunStepUpAuthIT.java
StepUpActionType.java | Resource | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/stepup/StepUpActionType.java
StepUpAuthController.java | Actor | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/stepup/StepUpAuthController.java
StepUpAuthService.java | Process | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/stepup/StepUpAuthService.java
StepUpAuthSteps.java | Resource | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/e2e/steps/StepUpAuthSteps.java
StepUpChallengeType.java | Resource | 6 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/stepup/StepUpChallengeType.java
StepUpCheckResponse.java | Resource | 6 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/stepup/dto/StepUpCheckResponse.java
StepUpHooks.java | Resource | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/e2e/hooks/StepUpHooks.java
StepUpRequestDto.java | Resource | 6 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/stepup/dto/StepUpRequestDto.java
StepUpTokenResponse.java | Resource | 6 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/stepup/dto/StepUpTokenResponse.java
```

## BE09 [EXECUTED] — What does AdminCascadeDeleteController cascade-delete?
rows=52 fp=e59611301dfd4c39
AUTO: Flow from AdminCascadeDeleteController.java, AdminCascadeDeleteService.java, AdminCascadeDeleteServiceImpl.java: 52 typed steps within 1 hops touching 25 downstream files (IMPORTS 21, INJECTS 12, MODIFIES 6, USES 6). Walk the gold rows in hop order.
```
AdminCascadeDeleteController.java | IMPORTS | AuthenticationTranslatableException.java | 7 | 1
AdminCascadeDeleteController.java | IMPORTS | RateLimit.java | 3 | 1
AdminCascadeDeleteController.java | IMPORTS | RateLimitKeyType.java | 3 | 1
AdminCascadeDeleteController.java | IMPORTS | RateLimitProfile.java | 3 | 1
AdminCascadeDeleteController.java | IMPORTS | ValidationTranslatableException.java | 7 | 1
AdminCascadeDeleteController.java | INJECTS | AdminCascadeDeleteService.java | 5 | 1
AdminCascadeDeleteController.java | PERFORMS | AdminCascadeDeleteService.java | 5 | 1
AdminCascadeDeleteService.java | IMPORTS | BatchCascadeDeletePreview.java | 5 | 1
AdminCascadeDeleteService.java | IMPORTS | CascadeDeletePreview.java | 5 | 1
AdminCascadeDeleteService.java | IMPORTS | CascadeDeleteResult.java | 5 | 1
```

## BE10 [EXECUTED] — Where is the Instagram OAuth callback handled?
rows=14 fp=d54057c2628e7f31
AUTO: Located: InstagramCallbackController.java (Actor, sub-5); InstagramConfig.java (Context, sub-5); InstagramConfigUnitTest.java (Rule, sub-8); InstagramDataDeletionService.java (Process, sub-5); InstagramDataDeletionServiceUnitTest.java (Rule, sub-5); InstagramDeauthorizationService.java (Process, sub-5) (+8 more). Paths in gold rows.
```
InstagramCallbackController.java | Actor | 5 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/controller/InstagramCallbackController.java
InstagramConfig.java | Context | 5 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/social/instagram/InstagramConfig.java
InstagramConfigUnitTest.java | Rule | 8 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/InstagramConfigUnitTest.java
InstagramDataDeletionService.java | Process | 5 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/service/InstagramDataDeletionService.java
InstagramDataDeletionServiceUnitTest.java | Rule | 5 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/InstagramDataDeletionServiceUnitTest.java
InstagramDeauthorizationService.java | Process | 5 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/service/InstagramDeauthorizationService.java
InstagramDeauthorizationServiceUnitTest.java | Rule | 5 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/InstagramDeauthorizationServiceUnitTest.java
InstagramPlatformApplication.java | Resource | 2 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/InstagramPlatformApplication.java
InstagramService.java | Process | 5 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/social/instagram/InstagramService.java
InstagramServiceUnitTest.java | Rule | 8 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/InstagramServiceUnitTest.java
```

## BE11 [EXECUTED] — How are legal documents and terms versions managed?
rows=11 fp=0bef51807a1326fb
AUTO: Located: LegalAdminController.java (Actor, sub-11); LegalController.java (Actor, sub-11); LegalDocument.java (Resource, sub-11); LegalDocumentDtoOut.java (Resource, sub-11); LegalDocumentRepository.java (Resource, sub-11); LegalDocumentService.java (Process, sub-11) (+5 more). Paths in gold rows.
```
LegalAdminController.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/LegalAdminController.java
LegalController.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/LegalController.java
LegalDocument.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/LegalDocument.java
LegalDocumentDtoOut.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/dto/LegalDocumentDtoOut.java
LegalDocumentRepository.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/LegalDocumentRepository.java
LegalDocumentService.java | Process | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/LegalDocumentService.java
LegalDocumentServiceUnitTest.java | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/LegalDocumentServiceUnitTest.java
LegalDocumentType.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/LegalDocumentType.java
TermsVersion.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/entity/TermsVersion.java
TermsVersionRepository.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/repository/TermsVersionRepository.java
```

## BE12 [EXECUTED] — Which languages does the backend support for messages and emails?
rows=1 fp=bdcc27cb9c58f291
AUTO: Located: AppLanguageFilter.java (Rule, sub-7). Paths in gold rows.
```
AppLanguageFilter.java | Rule | 7 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/translation/AppLanguageFilter.java
```

## BE13 [CONTENT_POINTER] — How do I set up the local database from scratch?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: checkItOut-be2/LOCAL_DATABASE_SETUP.md.

## BE14 [CONTENT_POINTER] — What did the 2026-09 pentest find and what was fixed?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: checkItOut-be2/docs/security/pentest-remediation-2026-09.md.

## BE15 [CONTENT_POINTER] — What is the storage cleanup strategy for uploaded files?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: checkItOut-be2/docs/STORAGE_CLEANUP_STRATEGY.md.

## BE16 [EXECUTED] — What happens end-to-end when a company upgrades its subscription plan?
rows=28 fp=4e99dbd200f3c6be
AUTO: Flow from SubscriptionService.java: 28 typed steps within 1 hops touching 12 downstream files (IMPORTS 7, INJECTS 7, USES 7, MODIFIES 6). Walk the gold rows in hop order.
```
SubscriptionService.java | IMPORTS | AppPaymentsProperties.java | 11 | 1
SubscriptionService.java | IMPORTS | NotificationType.java | 12 | 1
SubscriptionService.java | IMPORTS | PaymentsDisabledException.java | 11 | 1
SubscriptionService.java | IMPORTS | SubscriptionNotificationEvent.java | 12 | 1
SubscriptionService.java | IMPORTS | SubscriptionStatusDtoOut.java | 11 | 1
SubscriptionService.java | IMPORTS | User.java | 10 | 1
SubscriptionService.java | IMPORTS | UserRepository.java | 11 | 1
SubscriptionService.java | INJECTS | AppPaymentsProperties.java | 11 | 1
SubscriptionService.java | INJECTS | BillingPeriodRepository.java | 11 | 1
SubscriptionService.java | INJECTS | CompanySubscriptionRepository.java | 11 | 1
```

## BE17 [EXECUTED] — When invoice creation fails, how is it retried and when does it give up?
rows=23 fp=dba78f69c0a45436
AUTO: Flow from InvoiceRetryCronJob.java, InvoiceRetryService.java, InvoiceRetryServiceUnitTest.java: 23 typed steps within 1 hops touching 7 downstream files (IMPORTS 12, INJECTS 4, USES 3, MODIFIES 2). Walk the gold rows in hop order.
```
InvoiceRetryCronJob.java | IMPORTS | InvoiceRetryService.java | 11 | 1
InvoiceRetryCronJob.java | INJECTS | InvoiceRetryService.java | 11 | 1
InvoiceRetryCronJob.java | PERFORMS | InvoiceRetryService.java | 11 | 1
InvoiceRetryService.java | IMPORTS | CompanyData.java | 11 | 1
InvoiceRetryService.java | IMPORTS | CompanyDataRepository.java | 11 | 1
InvoiceRetryService.java | IMPORTS | InvoiceRecord.java | 11 | 1
InvoiceRetryService.java | IMPORTS | InvoiceRecordRepository.java | 11 | 1
InvoiceRetryService.java | IMPORTS | InvoiceStatus.java | 11 | 1
InvoiceRetryService.java | INJECTS | CompanyDataRepository.java | 11 | 1
InvoiceRetryService.java | INJECTS | InvoiceRecordRepository.java | 11 | 1
```

## BE18 [EXECUTED] — Which servlet filters run before a request reaches a controller, and in what order?
rows=12 fp=a913f9df6a5bdd6d
AUTO: Located: AppLanguageFilter.java (Rule, sub-7); BannedUserAuthorizationFilter.java (Rule, sub-9); ConsentEnforcementFilter.java (Rule, sub-11); CoopFilter.java (Rule, sub-4); CorsLoggingFilter.java (Rule, sub-7); EarlyRequestLoggingFilter.java (Rule, sub-7) (+6 more). Paths in gold rows. The ORDER is declared in FilterRegistrationConfiguration — content; the graph pins the set.
```
AppLanguageFilter.java | Rule | 7 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/translation/AppLanguageFilter.java
BannedUserAuthorizationFilter.java | Rule | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/authorization/BannedUserAuthorizationFilter.java
ConsentEnforcementFilter.java | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/authorization/ConsentEnforcementFilter.java
CoopFilter.java | Rule | 4 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/activecooperations/CoopFilter.java
CorsLoggingFilter.java | Rule | 7 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/logging/CorsLoggingFilter.java
EarlyRequestLoggingFilter.java | Rule | 7 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/logging/EarlyRequestLoggingFilter.java
EmailVerificationEnforcementFilter.java | Rule | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/authorization/EmailVerificationEnforcementFilter.java
FilterRegistrationConfiguration.java | Context | 7 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/logging/FilterRegistrationConfiguration.java
JwtAuthenticationFilter.java | Rule | 7 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/authorization/JwtAuthenticationFilter.java
RequestBodyCachingFilter.java | Rule | 7 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/logging/RequestBodyCachingFilter.java
```

## BE19 [EXECUTED] — How does a trial nearing expiry turn into a user-visible notification?
rows=31 fp=a09f37079d896cc5
AUTO: Flow from TrialExpiryNotifierCronJob.java: 31 typed steps within 2 hops touching 13 downstream files (IMPORTS 8, INJECTS 8, USES 7, MODIFIES 6). Walk the gold rows in hop order.
```
SubscriptionService.java | IMPORTS | AppPaymentsProperties.java | 11 | 2
SubscriptionService.java | IMPORTS | NotificationType.java | 12 | 2
SubscriptionService.java | IMPORTS | PaymentsDisabledException.java | 11 | 2
SubscriptionService.java | IMPORTS | SubscriptionNotificationEvent.java | 12 | 2
SubscriptionService.java | IMPORTS | SubscriptionStatusDtoOut.java | 11 | 2
SubscriptionService.java | IMPORTS | User.java | 10 | 2
SubscriptionService.java | IMPORTS | UserRepository.java | 11 | 2
SubscriptionService.java | INJECTS | AppPaymentsProperties.java | 11 | 2
SubscriptionService.java | INJECTS | BillingPeriodRepository.java | 11 | 2
SubscriptionService.java | INJECTS | CompanySubscriptionRepository.java | 11 | 2
```

## BE20 [EXECUTED] — What prevents two app instances from running the same cron job twice?
rows=9 fp=d2e5125064fb24b0
AUTO: Located: AnonymousConsentCleanupCronJob.java (Actor, sub-11); ConsentEnforcementCronJob.java (Actor, sub-11); DeferredDeletionCronJob.java (Actor, sub-5); EmailCronJob.java (Actor, sub-12); InvoiceRetryCronJob.java (Actor, sub-11); NoConsentAccountCleanupCronJob.java (Actor, sub-11) (+3 more). Paths in gold rows.
```
AnonymousConsentCleanupCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/AnonymousConsentCleanupCronJob.java
ConsentEnforcementCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/ConsentEnforcementCronJob.java
DeferredDeletionCronJob.java | Actor | 5 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/cron/DeferredDeletionCronJob.java
EmailCronJob.java | Actor | 12 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/notification/email/EmailCronJob.java
InvoiceRetryCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/cron/InvoiceRetryCronJob.java
NoConsentAccountCleanupCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/legal/NoConsentAccountCleanupCronJob.java
SubscriptionPeriodProcessorCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/cron/SubscriptionPeriodProcessorCronJob.java
TermsGraceProcessorCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/cron/TermsGraceProcessorCronJob.java
TrialExpiryNotifierCronJob.java | Actor | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/cron/TrialExpiryNotifierCronJob.java
```

## BE21 [EXECUTED] — How does publishing a new terms version force existing users to re-consent?
rows=34 fp=db7caf1fbb29f6d6
AUTO: Flow from TermsGraceProcessorCronJob.java: 34 typed steps within 2 hops touching 13 downstream files (IMPORTS 9, INJECTS 9, USES 7, MODIFIES 6). Walk the gold rows in hop order.
```
SubscriptionService.java | IMPORTS | AppPaymentsProperties.java | 11 | 2
SubscriptionService.java | IMPORTS | NotificationType.java | 12 | 2
SubscriptionService.java | IMPORTS | PaymentsDisabledException.java | 11 | 2
SubscriptionService.java | IMPORTS | SubscriptionNotificationEvent.java | 12 | 2
SubscriptionService.java | IMPORTS | SubscriptionStatusDtoOut.java | 11 | 2
SubscriptionService.java | IMPORTS | User.java | 10 | 2
SubscriptionService.java | IMPORTS | UserRepository.java | 11 | 2
SubscriptionService.java | INJECTS | AppPaymentsProperties.java | 11 | 2
SubscriptionService.java | INJECTS | BillingPeriodRepository.java | 11 | 2
SubscriptionService.java | INJECTS | CompanySubscriptionRepository.java | 11 | 2
```

## BE22 [EXECUTED] — Which features degrade if Redis is unavailable?
rows=12 fp=0edc2dc4ba094f70
AUTO: Located: NoOpRedisHealthIndicator.java (Resource, sub-3); RedisConfiguration.java (Context, sub-3); RedisGeoLocationCache.java (Resource, sub-3); RedisRateLimiterService.java (Process, sub-3); RedisServiceUnitTest.java (Rule, sub-3); RedisStartupConnectivityTest.java (Rule, sub-3) (+6 more). Paths in gold rows.
```
NoOpRedisHealthIndicator.java | Resource | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/health/NoOpRedisHealthIndicator.java
RedisConfiguration.java | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/config/RedisConfiguration.java
RedisGeoLocationCache.java | Resource | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/security/geoip/RedisGeoLocationCache.java
RedisRateLimiterService.java | Process | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/ratelimit/RedisRateLimiterService.java
RedisServiceUnitTest.java | Rule | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/RedisServiceUnitTest.java
RedisStartupConnectivityTest.java | Rule | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/config/RedisStartupConnectivityTest.java
RedisUserCache.java | Resource | 6 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/cache/RedisUserCache.java
RedisUserCacheUnitTest.java | Rule | 8 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/RedisUserCacheUnitTest.java
RedisValidationService.java | Process | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/redis/RedisValidationService.java
UserCacheService.java | Process | 6 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/auth/cache/UserCacheService.java
```

## BE23 [EXECUTED] — How does banning a user actually cut off their API access?
rows=4 fp=44e292a880012b0e
AUTO: Flow from BannedUserAuthorizationFilter.java: 4 typed steps within 1 hops touching 2 downstream files (IMPORTS 2, CONSTRAINS 1, INJECTS 1). Walk the gold rows in hop order.
```
BannedUserAuthorizationFilter.java | CONSTRAINS | UserCacheService.java | 6 | 1
BannedUserAuthorizationFilter.java | IMPORTS | BaseExceptionHandler.java | 7 | 1
BannedUserAuthorizationFilter.java | IMPORTS | UserCacheService.java | 6 | 1
BannedUserAuthorizationFilter.java | INJECTS | UserCacheService.java | 6 | 1
```

## BE24 [EXECUTED] — How is the number of campaigns a company can create limited by its plan?
rows=11 fp=68124da16a6d0cd0
AUTO: Flow from CampaignLimitService.java: 11 typed steps within 1 hops touching 4 downstream files (IMPORTS 3, INJECTS 3, MODIFIES 2, USES 2). Walk the gold rows in hop order.
```
CampaignLimitService.java | CALLS | SubscriptionService.java | 11 | 1
CampaignLimitService.java | IMPORTS | BillingPeriodRepository.java | 11 | 1
CampaignLimitService.java | IMPORTS | CompanySubscription.java | 11 | 1
CampaignLimitService.java | IMPORTS | CompanySubscriptionRepository.java | 11 | 1
CampaignLimitService.java | INJECTS | BillingPeriodRepository.java | 11 | 1
CampaignLimitService.java | INJECTS | CompanySubscriptionRepository.java | 11 | 1
CampaignLimitService.java | INJECTS | SubscriptionService.java | 11 | 1
CampaignLimitService.java | MODIFIES | BillingPeriodRepository.java | 11 | 1
CampaignLimitService.java | MODIFIES | CompanySubscriptionRepository.java | 11 | 1
CampaignLimitService.java | USES | BillingPeriodRepository.java | 11 | 1
```

## BE25 [EXECUTED] — How are GDPR account deletion requests executed and with what delay?
rows=41 fp=cd9d2ae30efa50ea
AUTO: Flow from DeferredDeletionCronJob.java, DeferredDeletionCronJobUnitTest.java: 41 typed steps within 2 hops touching 20 downstream files (IMPORTS 30, INJECTS 7, ACCESSES 2, EXTENDS 1). Walk the gold rows in hop order.
```
DeferredDeletionCronJob.java | ACCESSES | PendingDataDeletionRequestRepository.java | 5 | 1
DeferredDeletionCronJob.java | ACCESSES | UserAccountOrchestrator.java | 5 | 1
DeferredDeletionCronJob.java | IMPORTS | DeletionEligibilityDto.java | 5 | 1
DeferredDeletionCronJob.java | IMPORTS | DeletionRequestStatus.java | 5 | 1
DeferredDeletionCronJob.java | IMPORTS | PendingDataDeletionRequest.java | 5 | 1
DeferredDeletionCronJob.java | IMPORTS | PendingDataDeletionRequestRepository.java | 5 | 1
DeferredDeletionCronJob.java | IMPORTS | User.java | 10 | 1
DeferredDeletionCronJob.java | IMPORTS | UserAccountOrchestrator.java | 5 | 1
DeferredDeletionCronJob.java | INJECTS | PendingDataDeletionRequestRepository.java | 5 | 1
DeferredDeletionCronJob.java | INJECTS | UserAccountOrchestrator.java | 5 | 1
```

## BE26 [EXECUTED] — Who consumes GeoIP data and how does it interact with GDPR consent?
rows=16 fp=26b5261c1af2ca22
AUTO: Located: AdminGeoIpSteps.java (Resource, sub-3); GeoIpAdminController.java (Actor, sub-3); GeoIpConfiguration.java (Context, sub-3); GeoIpServiceUnitTest.java (Rule, sub-3); GeoIpStorageService.java (Process, sub-3); GeoLocation.java (Resource, sub-3) (+10 more). Paths in gold rows.
```
AdminGeoIpSteps.java | Resource | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/e2e/steps/AdminGeoIpSteps.java
GeoIpAdminController.java | Actor | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/admin/GeoIpAdminController.java
GeoIpConfiguration.java | Context | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/config/GeoIpConfiguration.java
GeoIpServiceUnitTest.java | Rule | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/GeoIpServiceUnitTest.java
GeoIpStorageService.java | Process | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/security/GeoIpStorageService.java
GeoLocation.java | Resource | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/security/GeoLocation.java
GeoLocationCache.java | Resource | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/security/geoip/GeoLocationCache.java
GeoLocationFacade.java | Actor | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/security/geoip/GeoLocationFacade.java
GeoLocationGdprController.java | Actor | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/security/GeoLocationGdprController.java
GeoLocationGdprService.java | Process | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/security/GeoLocationGdprService.java
```

## BE27 [EXECUTED] — How does an influencer's application changing status generate a notification?
rows=20 fp=242b808fb5b4a28a
AUTO: Located: AppliedOpportunityContentService.java (Process, sub-4); AppliedOpportunityContentServiceUnitTest.java (Rule, sub-13); AppliedOpportunityService.java (Process, sub-4); AppliedOpportunityServiceIntegrationTestBase.java (Resource, sub-13); AppliedOpportunityServiceUnitTest.java (Rule, sub-13); AppliedOpportunityService_Apply_IntegrationTest.java (Rule, sub-13) (+14 more). Paths in gold rows.
```
AppliedOpportunityContentService.java | Process | 4 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/appliedopportunities/AppliedOpportunityContentService.java
AppliedOpportunityContentServiceUnitTest.java | Rule | 13 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/AppliedOpportunityContentServiceUnitTest.java
AppliedOpportunityService.java | Process | 4 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/appliedopportunities/AppliedOpportunityService.java
AppliedOpportunityServiceIntegrationTestBase.java | Resource | 13 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/integration/service/appliedopportunity/AppliedOpportunityServiceIntegrationTestBase.java
AppliedOpportunityServiceUnitTest.java | Rule | 13 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/AppliedOpportunityServiceUnitTest.java
AppliedOpportunityService_Apply_IntegrationTest.java | Rule | 13 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/integration/service/appliedopportunity/AppliedOpportunityService_Apply_IntegrationTest.java
AppliedOpportunityService_Delete_IntegrationTest.java | Rule | 13 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/integration/service/appliedopportunity/AppliedOpportunityService_Delete_IntegrationTest.java
AppliedOpportunityService_FindById_IntegrationTest.java | Rule | 13 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/integration/service/appliedopportunity/AppliedOpportunityService_FindById_IntegrationTest.java
AppliedOpportunityService_FollowerValidation_IntegrationTest.java | Rule | 13 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/integration/service/appliedopportunity/AppliedOpportunityService_FollowerValidation_IntegrationTest.java
AppliedOpportunityService_Patch_IntegrationTest.java | Rule | 13 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/integration/service/appliedopportunity/AppliedOpportunityService_Patch_IntegrationTest.java
```

## BE28 [EXECUTED] — How is the language chosen for a transactional email sent by a cron job?
rows=2 fp=daf30d93162e278e
AUTO: Located: AppLanguageFilter.java (Rule, sub-7); EmailCronJob.java (Actor, sub-12). Paths in gold rows.
```
AppLanguageFilter.java | Rule | 7 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/translation/AppLanguageFilter.java
EmailCronJob.java | Actor | 12 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/notification/email/EmailCronJob.java
```

## BE29 [COVERAGE_GAP] — Which files in the repo hold credentials that must never reach a public release?
rows=0 fp=4f53cda18c2baa0c
AUTO: NOT IN GRAPH — no node matches; see gold_status.

## BE30 [EXECUTED] — Why can the backend refuse to start right after restoring a production database dump?
rows=2 fp=9b61514ef5bfab34
AUTO: Located: PaymentsDisabledBootGuard.java (Rule, sub-11); PaymentsDisabledBootGuardUnitTest.java (Rule, sub-11). Paths in gold rows.
```
PaymentsDisabledBootGuard.java | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/config/PaymentsDisabledBootGuard.java
PaymentsDisabledBootGuardUnitTest.java | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/subscription/PaymentsDisabledBootGuardUnitTest.java
```

## BE31 [EXECUTED] — Why would a company suddenly be unable to create new campaigns though they changed nothing?
rows=19 fp=6eea6dd543add338
AUTO: Flow from CampaignLimitExceededException.java, CampaignLimitService.java, CampaignLimitServiceUnitTest.java: 19 typed steps within 1 hops touching 6 downstream files (IMPORTS 10, INJECTS 3, MODIFIES 2, USES 2). Walk the gold rows in hop order.
```
CampaignLimitExceededException.java | EXTENDS | BusinessRuleViolationException.java | 7 | 1
CampaignLimitExceededException.java | IMPORTS | BusinessRuleViolationException.java | 7 | 1
CampaignLimitService.java | CALLS | SubscriptionService.java | 11 | 1
CampaignLimitService.java | IMPORTS | BillingPeriodRepository.java | 11 | 1
CampaignLimitService.java | IMPORTS | CompanySubscription.java | 11 | 1
CampaignLimitService.java | IMPORTS | CompanySubscriptionRepository.java | 11 | 1
CampaignLimitService.java | INJECTS | BillingPeriodRepository.java | 11 | 1
CampaignLimitService.java | INJECTS | CompanySubscriptionRepository.java | 11 | 1
CampaignLimitService.java | INJECTS | SubscriptionService.java | 11 | 1
CampaignLimitService.java | MODIFIES | BillingPeriodRepository.java | 11 | 1
```

## BE34 [EXECUTED] — Why does an invoice appear in the accounting system days after the payment succeeded?
rows=3 fp=64368260c0773078
AUTO: Flow from InvoiceRetryCronJob.java: 3 typed steps within 1 hops touching 1 downstream files (IMPORTS 1, INJECTS 1, PERFORMS 1). Walk the gold rows in hop order.
```
InvoiceRetryCronJob.java | IMPORTS | InvoiceRetryService.java | 11 | 1
InvoiceRetryCronJob.java | INJECTS | InvoiceRetryService.java | 11 | 1
InvoiceRetryCronJob.java | PERFORMS | InvoiceRetryService.java | 11 | 1
```

## BE35 [CONTENT_POINTER] — Why can saving the same record from two admin tabs make one save fail with a conflict?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: entity classes (all @Version).

## BE36 [EXECUTED] — How do I find every log line belonging to one failing user action?
rows=1 fp=546f1ebb76e37172
AUTO: Located: RequestCorrelationFilter.java (Rule, sub-7). Paths in gold rows.
```
RequestCorrelationFilter.java | Rule | 7 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/common/logging/RequestCorrelationFilter.java
```

## BE37 [EXECUTED] — Why are attachment URLs from some hosts rejected even when the file itself is fine?
rows=1 fp=4d27b7951a83e864
AUTO: Located: StorageUrlValidator.java (Rule, sub-3). Paths in gold rows.
```
StorageUrlValidator.java | Rule | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/storage/service/StorageUrlValidator.java
```

## FE01 [EXECUTED] — Which file defines the full route table and every URL the app serves?
rows=1 fp=05888c79bdb5a547
AUTO: Located: app.routes.ts (Resource, sub-17). Paths in gold rows.
```
app.routes.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/app.routes.ts
```

## FE02 [EXECUTED] — Where is the sign-in screen component defined?
rows=4 fp=b66306938b5fb228
AUTO: Located: sign-in.component.html (Actor, sub-17); sign-in.component.scss (Resource, sub-17); sign-in.component.spec.ts (Rule, sub-17); sign-in.component.ts (Actor, sub-17). Paths in gold rows.
```
sign-in.component.html | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/auth/sign-in/sign-in.component.html
sign-in.component.scss | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/auth/sign-in/sign-in.component.scss
sign-in.component.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/auth/sign-in/sign-in.component.spec.ts
sign-in.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/auth/sign-in/sign-in.component.ts
```

## FE03 [CONTENT_POINTER] — How do I add a new translated string and which files must stay in sync?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: src/assets/i18n/{en,pl}.json + check:i18n-parity gate.

## FE04 [CONTENT_POINTER] — Which languages does the app support and which is the default?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: app.config.ts (Transloco: en/pl, defaultLang pl).

## FE05 [CONTENT_POINTER] — Why does the app come up in Polish even when my browser prefers English?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: app.config.ts iter-107 comment.

## FE06 [EXECUTED] — What happens when an API call returns 401 - where is the sign-in redirect implemented?
rows=7 fp=2325fa1ab94f7102
AUTO: Flow from error.interceptor.spec.ts, error.interceptor.ts: 7 typed steps within 1 hops touching 3 downstream files (INJECTS 3, CONSTRAINS 2, IMPORTS 1, TESTED_BY 1). Walk the gold rows in hop order.
```
error.interceptor.spec.ts | IMPORTS | api | 18 | 1
error.interceptor.spec.ts | INJECTS | auth-api.service.ts | 17 | 1
error.interceptor.ts | CONSTRAINS | auth-api.service.ts | 17 | 1
error.interceptor.ts | CONSTRAINS | session-state.service.ts | 17 | 1
error.interceptor.ts | INJECTS | auth-api.service.ts | 17 | 1
error.interceptor.ts | INJECTS | session-state.service.ts | 17 | 1
error.interceptor.ts | TESTED_BY | error.interceptor.spec.ts | 17 | 1
```

## FE07 [EXECUTED] — Which interceptor attaches the X-Step-Up-Token header and when does it fire?
rows=11 fp=8e3c4c4a2c0c9e0a
AUTO: Located: step-up-auth.feature (Rule, sub-9); step-up-context.spec.ts (Rule, sub-17); step-up-context.ts (Resource, sub-17); step-up-dialog.component.html (Actor, sub-17); step-up-dialog.component.spec.ts (Rule, sub-17); step-up-dialog.component.ts (Actor, sub-17) (+5 more). Paths in gold rows.
```
step-up-auth.feature | Rule | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/resources/features/step-up-auth.feature
step-up-context.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/step-up/step-up-context.spec.ts
step-up-context.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/step-up/step-up-context.ts
step-up-dialog.component.html | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/shared/components/step-up-dialog/step-up-dialog.component.html
step-up-dialog.component.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/shared/components/step-up-dialog/step-up-dialog.component.spec.ts
step-up-dialog.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/shared/components/step-up-dialog/step-up-dialog.component.ts
step-up-dialog.fixture.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/sandbox/fixtures/step-up-dialog.fixture.ts
step-up.interceptor.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/step-up.interceptor.spec.ts
step-up.interceptor.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/step-up.interceptor.ts
step-up.service.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/step-up/step-up.service.spec.ts
```

## FE10 [COVERAGE_GAP] — How does the client authenticate API requests if no token is stored in the browser?
rows=0 fp=4f53cda18c2baa0c
AUTO: NOT IN GRAPH — no node matches; see gold_status.

## FE11 [EXECUTED] — How is the generated API client's base path wired to the backend in dev?
rows=1 fp=8a632c0c89621c57
AUTO: Located: proxy.conf.js (Context, sub-17). Paths in gold rows.
```
proxy.conf.js | Context | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/proxy.conf.js
```

## FE12 [CONTENT_POINTER] — How do I regenerate the API client after a backend schema change?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: package.json scripts openapi:gen / openapi:cycle.

## FE13 [EXECUTED] — Which service and generated API does the NIP company-lookup onboarding screen use?
rows=10 fp=3d3a71ef5949dbc4
AUTO: Flow from company-setup.component.html, company-setup.component.spec.ts, company-setup.component.ts...: 10 typed steps within 1 hops touching 1 downstream files (IMPORTS 5, INJECTS 2, TESTED_BY 2, PERFORMS 1). Walk the gold rows in hop order.
```
company-setup.component.spec.ts | IMPORTS | api | 18 | 1
company-setup.component.ts | IMPORTS | api | 18 | 1
company-setup.component.ts | INJECTS | registry.service.ts | 17 | 1
company-setup.component.ts | PERFORMS | registry.service.ts | 17 | 1
company-setup.component.ts | TESTED_BY | company-setup.component.spec.ts | 17 | 1
company-setup.fixture.ts | IMPORTS | api | 18 | 1
registry.service.spec.ts | IMPORTS | api | 18 | 1
registry.service.spec.ts | INJECTS | registry.service.ts | 17 | 1
registry.service.ts | IMPORTS | api | 18 | 1
registry.service.ts | TESTED_BY | registry.service.spec.ts | 17 | 1
```

## FE14 [CONTENT_POINTER] — Which screens are public marketing pages that render without login?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: app.routes.ts public routes.

## FE15 [EXECUTED] — What is the technical-survey section and which chapter pages does it contain?
rows=5 fp=8796682dbe9e82e3
AUTO: Located: survey-card.component.ts (Actor, sub-17); survey-hub.component.ts (Actor, sub-17); survey-hub.fixture.ts (Resource, sub-17); survey-links.ts (Resource, sub-17); survey-run-button.ts (Resource, sub-17). Paths in gold rows.
```
survey-card.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/survey/ui/survey-card.component.ts
survey-hub.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/survey/survey-hub.component.ts
survey-hub.fixture.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/sandbox/fixtures/survey-hub.fixture.ts
survey-links.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/survey/ui/survey-links.ts
survey-run-button.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/survey/ui/survey-run-button.ts
```

## FE16 [CONTENT_POINTER] — Which legacy URLs are kept alive as redirects so old bookmarks and Stripe return links still work?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: app.routes.ts redirect entries.

## FE17 [EXECUTED] — Where are the user settings tabs defined and how would I add a new one?
rows=7 fp=d0529451c682df3a
AUTO: Located: security-settings.component.html (Actor, sub-17); security-settings.component.spec.ts (Rule, sub-17); security-settings.component.ts (Actor, sub-17); security-settings.fixture.ts (Resource, sub-17); settings-layout.component.spec.ts (Rule, sub-17); settings-layout.component.ts (Actor, sub-17) (+1 more). Paths in gold rows.
```
security-settings.component.html | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/settings/security-settings.component.html
security-settings.component.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/settings/security-settings.component.spec.ts
security-settings.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/settings/security-settings.component.ts
security-settings.fixture.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/sandbox/fixtures/security-settings.fixture.ts
settings-layout.component.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/settings/settings-layout.component.spec.ts
settings-layout.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/settings/settings-layout.component.ts
social-connections-settings.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/settings/social-connections-settings.component.ts
```

## FE18 [EXECUTED] — How is the browser tab title produced from route definitions, and what happens on language switch?
rows=2 fp=6162c15b34cdb2db
AUTO: Located: seo-title.strategy.spec.ts (Rule, sub-17); seo-title.strategy.ts (Process, sub-17). Paths in gold rows.
```
seo-title.strategy.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/i18n/seo-title.strategy.spec.ts
seo-title.strategy.ts | Process | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/i18n/seo-title.strategy.ts
```

## FE19 [EXECUTED] — How are the visual parity tests organized and what compares legacy against greenfield pixels?
rows=1 fp=a172d8446428e8c4
AUTO: Located: component-pairs.ts (Rule, sub-17). Paths in gold rows.
```
component-pairs.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/e2e-tests/visual-parity/component-pairs.ts
```

## FE20 [CONTENT_POINTER] — Which Playwright device profiles exist and what must be installed for the Safari ones?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: playwright.config.ts device projects.

## FE21 [EXECUTED] — How does the BDD tier turn Gherkin feature files into runnable Playwright specs?
rows=34 fp=bf82c6cba0ad3b9b
AUTO: Located: account-activation-e2e.feature (Rule, sub-12); admin-geoip-analysis.feature (Rule, sub-9); admin-inactive-flow-consolidated.feature (Rule, sub-9); admin-platform-management.feature (Rule, sub-9); admin-user-management.feature (Rule, sub-9); consent-lifecycle.feature (Rule, sub-11) (+28 more). Paths in gold rows.
```
account-activation-e2e.feature | Rule | 12 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/resources/features/notification/account-activation-e2e.feature
admin-geoip-analysis.feature | Rule | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/resources/features/admin-geoip-analysis.feature
admin-inactive-flow-consolidated.feature | Rule | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/resources/features/admin/admin-inactive-flow-consolidated.feature
admin-platform-management.feature | Rule | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/resources/features/admin-platform-management.feature
admin-user-management.feature | Rule | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/resources/features/admin-user-management.feature
consent-lifecycle.feature | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/resources/features/consent/consent-lifecycle.feature
consent-module.feature | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/resources/features/consent/consent-module.feature
email-enforcement-filter.feature | Rule | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/resources/features/email-enforcement-filter.feature
file-upload-signed-url.feature | Rule | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/resources/features/files/file-upload-signed-url.feature
influencer-verification-password.feature | Rule | 9 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/resources/features/influencer-verification-password.feature
```

## FE22 [EXECUTED] — How do tests prove the new frontend talks to the backend the same way the old one did?
rows=1 fp=9d4c977a03ace84d
AUTO: Located: canonicalize.ts (Rule, sub-17). Paths in gold rows.
```
canonicalize.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/e2e-tests/integration/_trace/canonicalize.ts
```

## FE23 [EXECUTED] — How do e2e tests obtain a real logged-in session, and what happens when Firebase secrets are absent?
rows=1 fp=ac7ebeddf32a8fe7
AUTO: Located: real-login.ts (Rule, sub-17). Paths in gold rows.
```
real-login.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/e2e-tests/_framework/real-login.ts
```

## FE24 [EXECUTED] — What is the __sandbox route and why does ui-component-samples redirect there?
rows=7 fp=ddf3f28438e4fa2e
AUTO: Located: StripeService_Sandbox_IntegrationTest.java (Rule, sub-11); sandbox-director.service.spec.ts (Rule, sub-17); sandbox-director.service.ts (Process, sub-17); sandbox-host.component.ts (Actor, sub-17); sandbox-index.component.ts (Actor, sub-17); sandbox-registry.ts (Resource, sub-17) (+1 more). Paths in gold rows.
```
StripeService_Sandbox_IntegrationTest.java | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/integration/service/subscription/StripeService_Sandbox_IntegrationTest.java
sandbox-director.service.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/demo/sandbox-director.service.spec.ts
sandbox-director.service.ts | Process | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/demo/sandbox-director.service.ts
sandbox-host.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/sandbox/sandbox-host.component.ts
sandbox-index.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/sandbox/sandbox-index.component.ts
sandbox-registry.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/sandbox/sandbox-registry.ts
sandbox.routes.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/sandbox/sandbox.routes.ts
```

## FE25 [EXECUTED] — Where does the demo build intercept API calls and serve fixtures instead of hitting the backend?
rows=4 fp=4a1827b15105fae1
AUTO: Located: demo-fixtures.spec.ts (Rule, sub-17); demo-fixtures.ts (Resource, sub-17); demo.interceptor.ts (Rule, sub-17); scenario-registry.ts (Resource, sub-17). Paths in gold rows.
```
demo-fixtures.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/demo/demo-fixtures.spec.ts
demo-fixtures.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/demo/demo-fixtures.ts
demo.interceptor.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/demo/demo.interceptor.ts
scenario-registry.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/demo/scenario-registry.ts
```

## FE26 [EXECUTED] — How does the server-side auth probe see the browser session during SSR of guarded routes?
rows=1 fp=579f041ca7249aef
AUTO: Located: ssr-cookie-forward.interceptor.ts (Rule, sub-17). Paths in gold rows.
```
ssr-cookie-forward.interceptor.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/ssr-cookie-forward.interceptor.ts
```

## FE27 [CONTENT_POINTER] — Why are clicks made right after first paint not lost before the app becomes interactive?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: app.config.ts provideClientHydration(withEventReplay()).

## FE28 [EXECUTED] — Which response headers drive the consent and email-verification banners in the shell?
rows=4 fp=022e6d2152436aa4
AUTO: Flow from shell-headers.interceptor.spec.ts, shell-headers.interceptor.ts: 4 typed steps within 1 hops touching 1 downstream files (INJECTS 2, TESTED_BY 1, CONSTRAINS 1). Walk the gold rows in hop order.
```
shell-headers.interceptor.spec.ts | INJECTS | shell-status.service.ts | 17 | 1
shell-headers.interceptor.ts | CONSTRAINS | shell-status.service.ts | 17 | 1
shell-headers.interceptor.ts | INJECTS | shell-status.service.ts | 17 | 1
shell-headers.interceptor.ts | TESTED_BY | shell-headers.interceptor.spec.ts | 17 | 1
```

## FE29 [EXECUTED] — Where is 429 rate-limit handling implemented and which UI state does it feed?
rows=4 fp=32ed145addf03035
AUTO: Located: rate-limit-cache.interceptor.spec.ts (Rule, sub-17); rate-limit-cache.interceptor.ts (Rule, sub-17); rate-limit-state.service.ts (Process, sub-17); rate-limiting.feature (Rule, sub-3). Paths in gold rows.
```
rate-limit-cache.interceptor.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/rate-limit-cache.interceptor.spec.ts
rate-limit-cache.interceptor.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/rate-limit-cache.interceptor.ts
rate-limit-state.service.ts | Process | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/rate-limit/rate-limit-state.service.ts
rate-limiting.feature | Rule | 3 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/resources/features/rate-limiting.feature
```

## FE30 [CONTENT_POINTER] — What static checks run as part of check:full before a commit lands?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: package.json check:* scripts.

## FE31 [EXECUTED] — What does a user see when they type a URL that doesn't exist?
rows=5 fp=59acd8d4af463e01
AUTO: Located: auth-error.component.spec.ts (Rule, sub-17); auth-error.component.ts (Actor, sub-17); error-page.component.html (Actor, sub-17); error-page.component.spec.ts (Rule, sub-17); error-page.component.ts (Actor, sub-17). Paths in gold rows.
```
auth-error.component.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/auth/auth-error/auth-error.component.spec.ts
auth-error.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/auth/auth-error/auth-error.component.ts
error-page.component.html | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/error-page/error-page.component.html
error-page.component.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/error-page/error-page.component.spec.ts
error-page.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/error-page/error-page.component.ts
```

## FE32 [EXECUTED] — Which layout wraps authenticated pages versus auth pages, and where is the notification bell?
rows=10 fp=21df66d09b2d2d34
AUTO: Located: auth-layout.component.html (Actor, sub-17); auth-layout.component.scss (Resource, sub-17); auth-layout.component.spec.ts (Rule, sub-17); auth-layout.component.ts (Actor, sub-17); layout.component.html (Actor, sub-17); layout.component.scss (Resource, sub-17) (+4 more). Paths in gold rows.
```
auth-layout.component.html | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/layout/auth-layout/auth-layout.component.html
auth-layout.component.scss | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/layout/auth-layout/auth-layout.component.scss
auth-layout.component.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/layout/auth-layout/auth-layout.component.spec.ts
auth-layout.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/layout/auth-layout/auth-layout.component.ts
layout.component.html | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/layout/layout.component.html
layout.component.scss | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/layout/layout.component.scss
layout.component.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/layout/layout.component.spec.ts
layout.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/layout/layout.component.ts
settings-layout.component.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/settings/settings-layout.component.spec.ts
settings-layout.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/settings/settings-layout.component.ts
```

## FE33 [EXECUTED] — Why does the billing screen not use the freshly generated client for some payment and 2FA calls?
rows=2 fp=6a9445590987ab39
AUTO: Located: hidden-models.ts (Resource, sub-17); subscription.client.ts (Process, sub-17). Paths in gold rows.
```
hidden-models.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/api-frozen/hidden-models.ts
subscription.client.ts | Process | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/api-frozen/subscription.client.ts
```

## FE34 [EXECUTED] — What user-facing flows exist under collaborations, from browsing campaigns to reviewing submitted content?
rows=16 fp=a54c3c426828a9cb
AUTO: Located: CampaignLimitExceededException.java (Resource, sub-11); CampaignLimitService.java (Process, sub-11); CampaignLimitServiceUnitTest.java (Rule, sub-11); CampaignSnapshot.java (Resource, sub-4); campaign-applicants.component.html (Actor, sub-17); campaign-applicants.component.spec.ts (Rule, sub-17) (+10 more). Paths in gold rows.
```
CampaignLimitExceededException.java | Resource | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/CampaignLimitExceededException.java
CampaignLimitService.java | Process | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/subscription/CampaignLimitService.java
CampaignLimitServiceUnitTest.java | Rule | 11 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/test/java/com/sm/instagram/platform/unit/service/subscription/CampaignLimitServiceUnitTest.java
CampaignSnapshot.java | Resource | 4 | C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/main/java/com/sm/instagram/platform/notification/CampaignSnapshot.java
campaign-applicants.component.html | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/opportunities/campaign-applicants.component.html
campaign-applicants.component.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/opportunities/campaign-applicants.component.spec.ts
campaign-applicants.component.ts | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/opportunities/campaign-applicants.component.ts
campaign-applicants.fixture.ts | Resource | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/sandbox/fixtures/campaign-applicants.fixture.ts
collaboration-dashboard.component.html | Actor | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/collaborations/collaboration-dashboard.component.html
collaboration-dashboard.component.spec.ts | Rule | 17 | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/feature/collaborations/collaboration-dashboard.component.spec.ts
```

## GR02 [EXECUTED] — What breaks if AccountStatus.java changes?
rows=51 fp=8f58d2ed51937144
AUTO: 51 dependency edges from 49 files across 12 subsystems point at AccountStatus.java. Heaviest dependents: AccountActivatedEvent.java, AdminCheckRunner.java, AppliedOpportunityStatusHistoryServiceUnitTest.java, AuthControllerUnitTest.java, AuthService.java. Full sorted set in gold rows.
```
AccountActivatedEvent.java | AFFECTS | 12
AccountActivatedEvent.java | IMPORTS | 12
AccountActivatedEvent.java | INJECTS | 12
AdminCheckRunner.java | IMPORTS | 6
AppliedOpportunityStatusHistoryServiceUnitTest.java | IMPORTS | 13
AuthControllerUnitTest.java | IMPORTS | 6
AuthService.java | IMPORTS | 6
AuthServiceUnitTest.java | IMPORTS | 10
AuthorizationMoreUnitTest.java | IMPORTS | 7
AuthorizationServiceUnitTest.java | IMPORTS | 13
```

## GR03 [EXECUTED] — Which subsystems reach StripeService and over which edge types?
rows=11 fp=0926e63e16c024be
AUTO: 11 dependency edges from 8 files across 1 subsystems point at StripeService.java. Heaviest dependents: SubscriptionPaidController.java, SubscriptionServiceIntegrationTestBase.java, StripeServiceUnitTest.java, SubscriptionService_CronUnitTest.java, SubscriptionService_DowngradeUnitTest.java. Full sorted set in gold rows.
```
StripeServiceUnitTest.java | IMPORTS | 11
SubscriptionPaidController.java | IMPORTS | 11
SubscriptionPaidController.java | INJECTS | 11
SubscriptionPaidController.java | PERFORMS | 11
SubscriptionServiceIntegrationTestBase.java | IMPORTS | 11
SubscriptionServiceIntegrationTestBase.java | INJECTS | 11
SubscriptionService_CronUnitTest.java | IMPORTS | 11
SubscriptionService_DowngradeUnitTest.java | IMPORTS | 11
SubscriptionService_PaymentsToggleUnitTest.java | IMPORTS | 11
SubscriptionService_TermsUnitTest.java | IMPORTS | 11
```

## GR05 [EXECUTED] — Which files are boundary nodes between the greenfield FE (17) and the generated api (18)?
rows=156 fp=a03ac6b1d32e0d65
AUTO: Seam sub-17 <-> sub-18: 156 edges (IMPORTS 156). Beyond imports: none — the seam is pure imports.
```
account-deletion.fixture.ts | IMPORTS | api | 17->18
address.service.spec.ts | IMPORTS | api | 17->18
address.service.ts | IMPORTS | api | 17->18
addresses.component.spec.ts | IMPORTS | api | 17->18
addresses.component.ts | IMPORTS | api | 17->18
addresses.fixture.ts | IMPORTS | api | 17->18
admin-cascade-delete-dialog.component.spec.ts | IMPORTS | api | 17->18
admin-cascade-delete-dialog.component.ts | IMPORTS | api | 17->18
admin-cascade-delete.fixture.ts | IMPORTS | api | 17->18
admin-dictionary.fixture.ts | IMPORTS | api | 17->18
```

## GR06 [EXECUTED] — Where do the ALGEBRA_VIOLATION edges cluster - which subsystem pairs break the layer algebra?
rows=59 fp=13b96964a4809219
AUTO: 198 ALGEBRA_VIOLATION edges across 59 subsystem pairs. Hotspots: sub-4->sub-4 (29), sub-9->sub-9 (21), sub-11->sub-11 (17), sub-13->sub-4 (13), sub-3->sub-3 (10). These are typed-layer rule breaks — curation reviews the top pairs first.
```
0 | 2 | 1
0 | 4 | 1
1 | 1 | 2
1 | 11 | 1
1 | 4 | 5
10 | 1 | 1
10 | 10 | 1
10 | 11 | 1
10 | 9 | 1
11 | 11 | 17
```

## GR07 [EXECUTED] — Is dependency injection crossing subsystem boundaries where plain imports should suffice?
rows=82 fp=d1bf90f2063929cd
AUTO: 292 cross-subsystem INJECTS edges over 82 pairs — DI reaching across boundaries. Top: sub-13->sub-4 (13), sub-4->sub-11 (13), sub-9->sub-6 (13), sub-4->sub-8 (11), sub-9->sub-11 (11).
```
0 | 11 | 1
0 | 2 | 1
0 | 4 | 3
0 | 8 | 1
0 | 9 | 1
1 | 11 | 2
1 | 4 | 8
1 | 8 | 1
10 | 1 | 2
10 | 11 | 8
```

## GR09 [EXECUTED] — What are the entry points of subsystem 17 and which actor roots start it?
rows=5 fp=1c90967ccbc1989b
AUTO: Sub-17 entry: external in-edges point at nothing (a sink — use actor roots); actor roots: action-router.component.ts, addresses.component.html, addresses.component.ts, admin-cascade-delete-dialog.component.html, admin-cascade-delete-dialog.component.ts; spines (A_P_R, IDF-ordered) in gold rows.
```
ACTOR_ROOT | action-router.component.ts |  | 
ACTOR_ROOT | addresses.component.html |  | 
ACTOR_ROOT | addresses.component.ts |  | 
ACTOR_ROOT | admin-cascade-delete-dialog.component.html |  | 
ACTOR_ROOT | admin-cascade-delete-dialog.component.ts |  | 
```

## GR10 [EXECUTED] — Which controllers touch consent data and through which services?
rows=16 fp=dd10a0c31e884315
AUTO: Flow from ConsentAdminController.java, ConsentController.java, ConsentControllerUnitTest.java: 16 typed steps within 1 hops touching 6 downstream files (IMPORTS 12, INJECTS 2, PERFORMS 2). Walk the gold rows in hop order.
```
ConsentAdminController.java | IMPORTS | RateLimit.java | 3 | 1
ConsentAdminController.java | IMPORTS | RateLimitKeyType.java | 3 | 1
ConsentAdminController.java | IMPORTS | RateLimitProfile.java | 3 | 1
ConsentAdminController.java | IMPORTS | ResourceNotFoundException.java | 7 | 1
ConsentAdminController.java | IMPORTS | ValidationTranslatableException.java | 7 | 1
ConsentAdminController.java | INJECTS | ConsentService.java | 11 | 1
ConsentAdminController.java | PERFORMS | ConsentService.java | 11 | 1
ConsentController.java | IMPORTS | RateLimit.java | 3 | 1
ConsentController.java | IMPORTS | RateLimitKeyType.java | 3 | 1
ConsentController.java | IMPORTS | RateLimitProfile.java | 3 | 1
```

## GR12 [EXECUTED] — Which production subsystem does the test-only subsystem 13 validate?
rows=209 fp=af7fb9db29deac4c
AUTO: Seam sub-13 <-> sub-4: 209 edges (IMPORTS 189, INJECTS 14, EXTENDS 4, TESTED_BY 2). Beyond imports: AppliedOpportunityContentService.java TESTED_BY AppliedOpportunityContentServiceUnitTest.java; RepositoryResolver.java INJECTS PartnershipOpportunityPhotoRepository.java; PartnershipOpportunityPhotoRepository.java EXTENDS BaseRepository.java; PartnershipOpportunityPhotoRepository.java EXTENDS PartnershipOpportunityPhoto.java; PartnershipOpportunityService.java TESTED_BY PartnershipOpportunityServiceUnitTest.java.
```
ActiveCooperationControllerUnitTest.java | IMPORTS | OpportunityStatus.java | 13->4
ActiveCooperationControllerUnitTest.java | IMPORTS | RateStatus.java | 13->4
ActiveCooperationServiceIntegrationTestBase.java | IMPORTS | ActiveCooperationService.java | 13->4
ActiveCooperationServiceIntegrationTestBase.java | IMPORTS | AppliedOpportunity.java | 13->4
ActiveCooperationServiceIntegrationTestBase.java | IMPORTS | OpportunityStatus.java | 13->4
ActiveCooperationServiceIntegrationTestBase.java | IMPORTS | PartnershipOpportunity.java | 13->4
ActiveCooperationServiceIntegrationTestBase.java | IMPORTS | RateStatus.java | 13->4
ActiveCooperationServiceIntegrationTestBase.java | INJECTS | ActiveCooperationService.java | 13->4
ActiveCooperationService_Query_IntegrationTest.java | IMPORTS | AppliedOpportunity.java | 13->4
ActiveCooperationService_Query_IntegrationTest.java | IMPORTS | CoopDto.java | 13->4
```

## GR13 [EXECUTED] — Which production subsystems have no dedicated test coupling at all?
rows=19 fp=b3e9e90d8680722d
AUTO: Subsystems with zero TESTED_BY coupling: [0, 1, 2, 5, 7, 8, 9, 14, 16, 18]. Caveat: TESTED_BY edges (104 total) undercount real coverage — most test linkage rides IMPORTS from sub-13/sub-3; treat this as a shape signal, not a coverage report.
```
0 | 27 | NO_TEST_EDGES
1 | 18 | NO_TEST_EDGES
10 | 39 | tested
11 | 208 | tested
12 | 36 | tested
13 | 49 | tested
14 | 15 | NO_TEST_EDGES
15 | 35 | tested
16 | 12 | NO_TEST_EDGES
17 | 405 | tested
```

## GR15 [CONTENT_POINTER] — If we split MEGA subsystem 17, where is the natural cut line?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: dossier 17: v3_overlap {0:180, 1:148} + folders.

## GR16 [EXECUTED] — Which services act as hubs bridging multiple subsystems?
rows=52 fp=a9e8f7df687fbbc7
AUTO: 52 hyperedge hubs bridge multiple subsystems: FirebaseAuthProxyService.java (A_P_A, subs [9, 10]); LegalConsentService.java (A_P_A, subs [9, 11]); StepUpAuthService.java (A_P_A, subs [4, 9]); TokenExchangeService.java (A_P_A, subs [6, 10, 11]); UserCacheService.java (A_P_A, subs [6, 9, 11]); UserPreferencesService.java (A_P_A, subs [0, 9]).
```
A_P_A | FirebaseAuthProxyService.java | 2.862 | 9,10
A_P_A | LegalConsentService.java | 1.609 | 9,11
A_P_A | StepUpAuthService.java | 2.457 | 4,9
A_P_A | TokenExchangeService.java | 2.457 | 6,10,11
A_P_A | UserCacheService.java | 2.457 | 6,9,11
A_P_A | UserPreferencesService.java | 2.862 | 0,9
A_P_R | ActiveCooperationService.java | 2.197 | 4,11
A_P_R | AddressService.java | 1.638 | 1,4,8,11
A_P_R | AppliedOpportunityContentService.java | 1.504 | 4,11
A_P_R | AppliedOpportunityService.java | 1.504 | 2,4,8,9,11
```

## GR17 [EXECUTED] — What is the dependency direction between subscriptions (11) and logging/exceptions (7)?
rows=61 fp=fa2701a454161ede
AUTO: Seam sub-11 <-> sub-7: 61 edges (IMPORTS 51, INJECTS 4, VALIDATES 3, EXTENDS 3). Beyond imports: WebSecurityConfiguration.java INJECTS ConsentEnforcementFilter.java; WebSecurityConfiguration.java INJECTS CorsProperties.java; WebSecurityConfiguration.java VALIDATES CorsProperties.java; CorsLoggingFilter.java INJECTS CorsProperties.java; CorsLoggingFilter.java VALIDATES CorsProperties.java.
```
AuthorizationMoreUnitTest.java | IMPORTS | UserRepository.java | 7->11
AuthorizationMoreUnitTest.java | IMPORTS | UserType.java | 7->11
BusinessExceptionHandler.java | IMPORTS | PaymentsDisabledException.java | 7->11
CampaignLimitExceededException.java | EXTENDS | BusinessRuleViolationException.java | 11->7
CampaignLimitExceededException.java | IMPORTS | BusinessRuleViolationException.java | 11->7
CeidgRegistryAdapter.java | IMPORTS | ExternalServiceException.java | 11->7
CeidgRegistryAdapterUnitTest.java | IMPORTS | ExternalServiceException.java | 11->7
ConsentAdminController.java | IMPORTS | ResourceNotFoundException.java | 11->7
ConsentAdminController.java | IMPORTS | ValidationTranslatableException.java | 11->7
ConsentController.java | IMPORTS | ResourceNotFoundException.java | 11->7
```

## GR18 [EXECUTED] — Which subsystems implement authentication and how do they talk to each other?
rows=51 fp=046feeed5116d69f
AUTO: Seam sub-9 <-> sub-6: 51 edges (IMPORTS 25, INJECTS 15, CALLS 7, PERFORMS 2, CONSTRAINS 2). Beyond imports: TestAuthController.java INJECTS UserCacheService.java; TestAuthController.java PERFORMS UserCacheService.java; TestAuthController.java INJECTS FirestoreService.java; TestAuthController.java PERFORMS FirestoreService.java; AdminIntegrityChecker.java INJECTS TotpFirestoreService.java.
```
AdminIntegrityChecker.java | IMPORTS | TotpFirestoreService.java | 9->6
AdminIntegrityChecker.java | INJECTS | TotpFirestoreService.java | 9->6
BannedUserAuthorizationFilter.java | CONSTRAINS | UserCacheService.java | 9->6
BannedUserAuthorizationFilter.java | IMPORTS | UserCacheService.java | 9->6
BannedUserAuthorizationFilter.java | INJECTS | UserCacheService.java | 9->6
EmailChangeService.java | CALLS | UserCacheService.java | 9->6
EmailChangeService.java | IMPORTS | UserCacheService.java | 9->6
EmailChangeService.java | INJECTS | UserCacheService.java | 9->6
EmailVerificationEnforcementFilter.java | CONSTRAINS | UserCacheService.java | 9->6
EmailVerificationEnforcementFilter.java | IMPORTS | UserCacheService.java | 9->6
```

## GR19 [EXECUTED] — Over which edges can anything reach the TOTP/step-up subsystem?
rows=6 fp=a67669ecebc48535
AUTO: Sub-6 entry: external in-edges point at UserCacheService.java, FirestoreService.java, TotpFirestoreService.java; actor roots: none; spines (A_P_R, IDF-ordered) in gold rows.
```
ENTRY | FirestoreService.java | 25 | 
ENTRY | KMSValidationService.java | 4 | 
ENTRY | TotpFirestoreService.java | 23 | 
ENTRY | TwoFactorAuthService.java | 6 | 
ENTRY | UserCacheService.java | 51 | 
SPINE | RegistrationService.java | 2.197 | AuthController.java|PlatformRepository.java|UserRepository.java|UserSocialConnectionRepository.java
```

## GR20 [EXECUTED] — Which resources does the cascade-deletion subsystem modify, in what order?
rows=55 fp=6b4904c79e31d81c
AUTO: Flow from AdminCascadeDeleteController.java, AdminCascadeDeleteService.java, AdminCascadeDeleteServiceImpl.java...: 55 typed steps within 1 hops touching 21 downstream files (IMPORTS 22, INJECTS 12, MODIFIES 6, USES 6). Walk the gold rows in hop order.
```
AdminCascadeDeleteController.java | IMPORTS | AuthenticationTranslatableException.java | 7 | 1
AdminCascadeDeleteController.java | IMPORTS | RateLimit.java | 3 | 1
AdminCascadeDeleteController.java | IMPORTS | RateLimitKeyType.java | 3 | 1
AdminCascadeDeleteController.java | IMPORTS | RateLimitProfile.java | 3 | 1
AdminCascadeDeleteController.java | IMPORTS | ValidationTranslatableException.java | 7 | 1
AdminCascadeDeleteController.java | INJECTS | AdminCascadeDeleteService.java | 5 | 1
AdminCascadeDeleteController.java | PERFORMS | AdminCascadeDeleteService.java | 5 | 1
AdminCascadeDeleteService.java | IMPORTS | BatchCascadeDeletePreview.java | 5 | 1
AdminCascadeDeleteService.java | IMPORTS | CascadeDeletePreview.java | 5 | 1
AdminCascadeDeleteService.java | IMPORTS | CascadeDeleteResult.java | 5 | 1
```

## GR21 [EXECUTED] — Inside subsystem 11, do any controllers sit below services in trophic height (layer inversion)?
rows=3 fp=45d1192113c75f7a
AUTO: Sub-11 trophic check (MacKay heights on the internal digraph, gauged PER weakly-connected component; controllers are upstream/low when healthy). Compared within each component — service medians c0:0.97; inversions (controller above its OWN component's median): PublicConfigController.java, LegalAdminController.java, TestLegalController.java; controllers in Service-less components (test undefined): none.
```
LegalAdminController.java | 1.27 | CONTROLLER_ABOVE_SERVICE_MEDIAN
PublicConfigController.java | 1.09 | CONTROLLER_ABOVE_SERVICE_MEDIAN
TestLegalController.java | 1.07 | CONTROLLER_ABOVE_SERVICE_MEDIAN
```

## GR22 [EXECUTED] — Which Event nodes exist and who produces/consumes each?
rows=8 fp=e77243c10746dcbc
AUTO: 8 Event nodes exist. AccountActivatedEvent.java (sub-12; triggered by EmailVerificationService.java|RegistryLookupService.java|UserService.java); InvoiceCreatedEvent.java (sub-11; triggered by -); NewUserRegisteredEvent.java (sub-12; triggered by -); OpportunityStatusChangedEvent.java (sub-4; triggered by AppliedOpportunityContentService.java|AppliedOpportunityService.java); SubscriptionEvent.java (sub-11; triggered by -); SubscriptionNotificationEvent.java (sub-12; triggered by SubscriptionService.java); TestEnvironmentGuard.java (sub-3; triggered by -); TotpValidationService.java (sub-6; triggered by -). Caveat: TRIGGERS edges are sparse (6 system-wide) — consumers are under-modelled until delta enrichment.
```
AccountActivatedEvent.java | 12 | EmailVerificationService.java|RegistryLookupService.java|UserService.java | EmailVerificationService.java|RegistryLookupService.java|SubscriptionNotificationEventListener.java|UserService.java
InvoiceCreatedEvent.java | 11 | - | -
NewUserRegisteredEvent.java | 12 | - | -
OpportunityStatusChangedEvent.java | 4 | AppliedOpportunityContentService.java|AppliedOpportunityService.java | AppliedOpportunityContentService.java|AppliedOpportunityService.java
SubscriptionEvent.java | 11 | - | SubscriptionEventRepository.java
SubscriptionNotificationEvent.java | 12 | SubscriptionService.java | SubscriptionService.java|SubscriptionService_TermsUnitTest.java
TestEnvironmentGuard.java | 3 | - | CommonSafetyUnitTest.java
TotpValidationService.java | 6 | - | AuthValidatorsUnitTest.java|TotpValidationServiceUnitTest.java
```

## GR23 [EXECUTED] — Which config files influence the most subsystems?
rows=15 fp=f6515805200f3a11
AUTO: Highest-reach configuration by in-degree: CucumberSpringConfig.java (48 deps, sub-3), AppPaymentsProperties.java (20 deps, sub-11), RecaptchaConfig.java (14 deps, sub-8), RegistryProperties.java (13 deps, sub-11), CorsProperties.java (12 deps, sub-11), StripeProperties.java (10 deps, sub-11).
```
AppPaymentsConfiguration.java | 0 | 11
AppPaymentsProperties.java | 20 | 11
AsyncConfig.java | 0 | 10
CorsProperties.java | 12 | 11
CucumberSpringConfig.java | 48 | 3
EmailConfigurationProperties.java | 7 | 10
FakturowniaProperties.java | 1 | 11
InstagramConfig.java | 7 | 5
RateLimitProperties.java | 6 | 3
RecaptchaConfig.java | 14 | 8
```

## GR24 [EXECUTED] — Which single files would disconnect a subsystem if removed?
rows=5 fp=4a3931f520799250
AUTO: Single-file seam carriers (>=90% of a >=10-edge seam through one node): User.java carries 36/39 of 11->10; UserCacheService.java carries 14/14 of 11->6; User.java carries 18/18 of 12->10; User.java carries 14/14 of 13->10; api carries 156/156 of 17->18. Removing such a file disconnects the subsystems — structural single points of failure.
```
11->10 | User.java | 36 | 39
11->6 | UserCacheService.java | 14 | 14
12->10 | User.java | 18 | 18
13->10 | User.java | 14 | 14
17->18 | api | 156 | 156
```

## GR26 [CONTENT_POINTER] — Did subscription and consent code merge or separate across reindexes?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: dossier 11: v3_overlap {6:78, 7:72}.

## GR27 [EXECUTED] — What depends on UserSocialConnectionRepository and who owns that area?
rows=40 fp=1d93c372994a8984
AUTO: 40 dependency edges from 16 files across 7 subsystems point at UserSocialConnectionRepository.java. Heaviest dependents: AppliedOpportunityService.java, AuthService.java, InstagramDataDeletionService.java, InstagramDeauthorizationService.java, OAuthCallbackService.java. Full sorted set in gold rows.
```
AppliedOpportunityService.java | IMPORTS | 4
AppliedOpportunityService.java | INJECTS | 4
AppliedOpportunityService.java | MODIFIES | 4
AppliedOpportunityService.java | USES | 4
AppliedOpportunityServiceIntegrationTestBase.java | IMPORTS | 13
AppliedOpportunityServiceIntegrationTestBase.java | INJECTS | 13
AppliedOpportunityServiceUnitTest.java | IMPORTS | 13
AuthService.java | IMPORTS | 6
AuthService.java | INJECTS | 6
AuthService.java | MODIFIES | 6
```

## GR29 [EXECUTED] — Which files have the highest in-degree system-wide?
rows=15 fp=1bbb9fd69a385d6d
AUTO: Most load-bearing files by dependency in-degree: UserRepository.java (184, sub-11), User.java (161, sub-10), api (156, sub-18), ResourceNotFoundException.java (101, sub-7), ValidationTranslatableException.java (96, sub-7), PermissionUtils.java (95, sub-4), BaseRepository.java (70, sub-4), PartnershipOpportunity.java (60, sub-4).
```
AccountStatus.java | 51 | 16
BaseRepository.java | 70 | 4
CucumberSpringConfig.java | 48 | 3
InsufficientPermissionsException.java | 58 | 7
PartnershipOpportunity.java | 60 | 4
PermissionUtils.java | 95 | 4
RepositoryResolver.java | 50 | 4
ResourceNotFoundException.java | 101 | 7
SpecificationBuilder.java | 49 | 8
User.java | 161 | 10
```

## GR30 [EXECUTED] — Which subsystems would a CORS configuration change touch?
rows=12 fp=78313c356851c5dd
AUTO: 12 dependency edges from 6 files across 1 subsystems point at CorsProperties.java. Heaviest dependents: CorsLoggingFilter.java, EarlyRequestLoggingFilter.java, WebSecurityConfiguration.java, CorsLoggingFilterUnitTest.java, FilterRegistrationConfiguration.java. Full sorted set in gold rows.
```
CorsLoggingFilter.java | IMPORTS | 7
CorsLoggingFilter.java | INJECTS | 7
CorsLoggingFilter.java | VALIDATES | 7
CorsLoggingFilterUnitTest.java | IMPORTS | 7
EarlyRequestLoggingFilter.java | IMPORTS | 7
EarlyRequestLoggingFilter.java | INJECTS | 7
EarlyRequestLoggingFilter.java | VALIDATES | 7
FilterRegistrationConfiguration.java | IMPORTS | 7
RequestBodyCachingFilterUnitTest.java | IMPORTS | 7
WebSecurityConfiguration.java | IMPORTS | 7
```

## GR31 [EXECUTED] — Which resources are shared by multiple processes (contention points)?
rows=21 fp=39557a89723d18f4
AUTO: 21 hyperedge cohort(s): P_R_P hub AddressRepository.java (idf 1.658) with 4 members; P_R_P hub AppliedOpportunityContentRepository.java (idf 2.351) with 2 members; P_R_P hub AppliedOpportunityRepository.java (idf 1.253) with 6 members; P_R_P hub AppliedOpportunityStatusHistoryRepository.java (idf 2.351) with 2 members; P_R_P hub BaseRepository.java (idf 1.946) with 3 members. Members co-change with the hub; IDF weights distinctiveness.
```
P_R_P | AddressRepository.java | sub-1 | 1.658 | AddressService.java|AuthService.java|PartnershipOpportunityService.java|UserService.java
P_R_P | AppliedOpportunityContentRepository.java | sub-4 | 2.351 | AdminCascadeDeleteServiceImpl.java|AppliedOpportunityContentService.java
P_R_P | AppliedOpportunityRepository.java | sub-4 | 1.253 | ActiveCooperationService.java|AdminCascadeDeleteServiceImpl.java|AppliedOpportunityContentService.java|AppliedOpportunityService.java|AppliedOpportunityStatusHistoryService.java|PartnershipOpportunityService.java
P_R_P | AppliedOpportunityStatusHistoryRepository.java | sub-4 | 2.351 | AdminCascadeDeleteServiceImpl.java|AppliedOpportunityStatusHistoryService.java
P_R_P | BaseRepository.java | sub-4 | 1.946 | BaseService.java|ContentTypeService.java|PlatformService.java
P_R_P | BillingPeriodRepository.java | sub-11 | 2.351 | CampaignLimitService.java|SubscriptionService.java
P_R_P | CompanyDataRepository.java | sub-11 | 1.658 | EmailVerificationService.java|FirebaseAuthProxyService.java|InvoiceRetryService.java|RegistryLookupService.java
P_R_P | CompanySubscriptionRepository.java | sub-11 | 2.351 | CampaignLimitService.java|SubscriptionService.java
P_R_P | ConsentRecordRepository.java | sub-11 | 2.351 | LegalConsentService.java|TokenExchangeService.java
P_R_P | FaqCategoryRepository.java | sub-14 | 2.351 | FaqCategoryService.java|FaqService.java
```

## GR32 [CONTENT_POINTER] — Which sandbox fixtures in sub-17 exercise which backend subsystems?
rows=0 fp=4f53cda18c2baa0c
AUTO: Answer lives in content, not structure. Pointer: e2e-tests fixtures mirror BE contracts by name; no FE->BE edges modeled (separate repos).
