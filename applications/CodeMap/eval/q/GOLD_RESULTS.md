# Q1 gold — executed results digest

## M01 — What is the one-walk overview of every subsystem: name, size, entry point?
rows=19 fp=4bde1dc306dc55c9 subs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
note: rows: [sub, size, dominant_type, purity, top_entry(by external in-degree), ext_in]
```
0 | 27 | Rule | 0.63 | UserPreferences.java | 22
1 | 18 | Resource | 0.44 | AddressRepository.java | 24
10 | 39 | Rule | 0.49 | User.java | 154
11 | 208 | Resource | 0.53 | UserRepository.java | 141
12 | 36 | Resource | 0.44 | EmailFrequency.java | 8
13 | 49 | Rule | 0.90 | AppliedOpportunityServiceIntegrationTestBase.java | 3
14 | 15 | Resource | 0.67 | Faq.java | 3
15 | 35 | Resource | 0.69 | SupportTicket.java | 5
16 | 12 | Resource | 0.67 | AccountStatus.java | 51
17 | 405 | Actor | 0.40 | - | 0
18 | 1 | Resource | 1.00 | api | 156
2 | 15 | Resource | 0.73 | UserSocialConnectionRepository.java | 40
3 | 129 | Context | 0.27 | CucumberSpringConfig.java | 44
4 | 123 | Resource | 0.69 | PermissionUtils.java | 73
5 | 48 | Resource | 0.50 | InstagramService.java | 8
6 | 51 | Resource | 0.53 | UserCacheService.java | 51
7 | 52 | Resource | 0.52 | ResourceNotFoundException.java | 101
8 | 49 | Rule | 0.67 | SpecificationBuilder.java | 48
9 | 103 | Rule | 0.50 | UserPreferencesRepository.java | 33
```

## M02 — What is the minimal reading path to understand the subscription/consent subsystem?
rows=6 fp=78065ed1ad7eaf41 subs=[11]
note: v1 spine definition: entry = max external in-degree member of sub-11; spines = top-IDF A_P_R hyperedges hubbed in sub-11 (C3 will refine with trophic order)
```
ENTRY | UserRepository.java | 141 | 
SPINE | CampaignLimitService.java | 2.485 | BillingPeriodRepository.java|CompanySubscriptionRepository.java|TestSubscriptionController.java
SPINE | ConsentService.java | 1.504 | PermissionUtils.java|ConsentAdminController.java|ConsentController.java|ConsentDefinitionRepository.java|ConsentVersionRepository.java|UserConsentRepository.java|UserCurrentConsentRepository.java|UserRepository.java
SPINE | InvoiceRetryService.java | 1.974 | CompanyDataRepository.java|InvoiceRetryCronJob.java|InvoicingPort.java|InvoiceRecordRepository.java|TestSubscriptionController.java
SPINE | LegalDocumentService.java | 2.485 | LegalController.java|LegalDocumentRepository.java|TestLegalController.java
SPINE | StripeService.java | 2.89 | StripeProperties.java|SubscriptionPaidController.java
```

## M03 — If UserRepository.java changes, which files and subsystems are affected?
rows=184 fp=1fba123ea1b43f02 subs=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
```
ActiveCooperationService.java | IMPORTS | 4
ActiveCooperationService.java | INJECTS | 4
ActiveCooperationService.java | MODIFIES | 4
ActiveCooperationService.java | USES | 4
AddressMapping.java | IMPORTS | 1
AddressMapping.java | INJECTS | 1
AddressService.java | IMPORTS | 1
AddressService.java | INJECTS | 1
AddressService.java | MODIFIES | 1
AddressService.java | USES | 1
AddressServiceUnitTest.java | IMPORTS | 1
AdminCascadeDeleteServiceImpl.java | IMPORTS | 5
AdminCascadeDeleteServiceImpl.java | INJECTS | 5
AdminCascadeDeleteServiceImpl.java | MODIFIES | 5
AdminCascadeDeleteServiceImpl.java | USES | 5
AdminCheckRunner.java | IMPORTS | 6
AdminCheckRunner.java | INJECTS | 6
AdminIntegrityChecker.java | IMPORTS | 9
AdminIntegrityChecker.java | INJECTS | 9
AppliedOpportunityContentService.java | IMPORTS | 4
... (+164 more)
```

## M04 — Why is a user with a valid login token still getting 403s on every request?
rows=4 fp=5f6546a7923f95df subs=[9, 11]
```
BannedUserAuthorizationFilter.java | Rule | 9
ConsentEnforcementFilter.java | Rule | 11
EmailVerificationEnforcementFilter.java | Rule | 9
EmailVerificationService.java | Process | 9
```

## M05 — Why do some accounts vanish a few days after registration without any admin action?
rows=2 fp=f9c02988d2bece59 subs=[11]
```
INJECTS | LegalConsentService.java | 11
PERFORMS | LegalConsentService.java | 11
```

## M06 — Why do multi-select relation fields arrive at the backend as an empty object?
rows=2 fp=a37501a6d989a481 subs=[17]
```
set-to-array.interceptor.spec.ts | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/set-to-array.interceptor.spec.ts | 
set-to-array.interceptor.ts | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/set-to-array.interceptor.ts | 
```

## M07 — Which notification/email flows are triggered from the subscription lifecycle?
rows=50 fp=e5232f060e9cd98c subs=[10, 11, 12]
```
BillingPeriod.java | IMPORTS | User.java | 10
CampaignLimitServiceUnitTest.java | IMPORTS | User.java | 10
CompanyData.java | IMPORTS | User.java | 10
CompanySubscription.java | IMPORTS | User.java | 10
ConsentRecord.java | IMPORTS | User.java | 10
ConsentService.java | IMPORTS | User.java | 10
ConsentServiceUnitTest.java | IMPORTS | User.java | 10
InvoiceRecord.java | IMPORTS | User.java | 10
InvoiceRetryServiceUnitTest.java | IMPORTS | User.java | 10
LegalConsentService.java | IMPORTS | User.java | 10
LegalConsentServiceUnitTest.java | IMPORTS | User.java | 10
LegalController.java | IMPORTS | TokenExchangeService.java | 10
LegalController.java | IMPORTS | User.java | 10
LegalController.java | INJECTS | TokenExchangeService.java | 10
LegalController.java | PERFORMS | TokenExchangeService.java | 10
RegistryLookupService.java | IMPORTS | AccountActivatedEvent.java | 12
RegistryLookupService.java | IMPORTS | User.java | 10
RegistryLookupService.java | TRIGGERS | AccountActivatedEvent.java | 12
RegistryLookupServiceUnitTest.java | IMPORTS | User.java | 10
RegistryLookupService_Confirm_IntegrationTest.java | IMPORTS | User.java | 10
... (+30 more)
```

## M08 — What can reach payment resources from a public controller within 3 hops?
rows=3 fp=ffb87c839fce8be6 subs=[4, 11]
note: 'public' approximated as any Actor outside sub-11; endpoint-level auth annotation is a C4 enrichment
```
PartnershipOpportunityController.java | 4 | CompanySubscription.java | 3
PartnershipOpportunityController.java | 4 | CompanySubscriptionRepository.java | 3
PartnershipOpportunityController.java | 4 | SubscriptionService.java | 3
```

## M09 — Which files co-participate in hyperedges with address.service.ts?
rows=1 fp=75faf07ecd53bd4f subs=[17]
```
A_P_A | address.service.ts | 2.862 | addresses.component.ts|layout.component.ts
```

## M10 — In what order do the HTTP interceptors run and why does the order matter?
rows=9 fp=d49d7b93dfa5613a subs=[17]
note: the ORDER itself is content, not structure: recipe returns the interceptor set + app.config.ts as the read pointer
```
app.config.ts | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/app.config.ts
demo.interceptor.ts | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/demo/demo.interceptor.ts
error.interceptor.ts | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/error.interceptor.ts
language.interceptor.ts | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/language.interceptor.ts
rate-limit-cache.interceptor.ts | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/rate-limit-cache.interceptor.ts
set-to-array.interceptor.ts | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/set-to-array.interceptor.ts
shell-headers.interceptor.ts | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/shell-headers.interceptor.ts
ssr-cookie-forward.interceptor.ts | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/ssr-cookie-forward.interceptor.ts
step-up.interceptor.ts | C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/src/app/core/interceptors/step-up.interceptor.ts
```

## M11 — Which cron jobs can modify user data, and via which paths?
rows=10 fp=51d30e13803c1206 subs=[11]
```
AnonymousConsentCleanupCronJob.java | - | -
ConsentEnforcementCronJob.java | - | -
DeferredDeletionCronJob.java | ACCESSES | PendingDataDeletionRequestRepository.java
DeferredDeletionCronJob.java | ACCESSES | UserAccountOrchestrator.java
EmailCronJob.java | - | -
InvoiceRetryCronJob.java | - | -
NoConsentAccountCleanupCronJob.java | - | -
SubscriptionPeriodProcessorCronJob.java | ACCESSES | AppPaymentsProperties.java
TermsGraceProcessorCronJob.java | ACCESSES | AppPaymentsProperties.java
TrialExpiryNotifierCronJob.java | - | -
```

## M12 — Which subsystem has the highest cross-boundary coupling?
rows=19 fp=e3b51f21e941911f subs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
note: rows: [sub, size, internal_edges, external_edges, external_ratio]
```
0 | 27 | 10 | 157 | 0.940
1 | 18 | 33 | 161 | 0.830
10 | 39 | 43 | 464 | 0.915
11 | 208 | 463 | 480 | 0.509
12 | 36 | 55 | 113 | 0.673
13 | 49 | 23 | 349 | 0.938
14 | 15 | 60 | 58 | 0.492
15 | 35 | 82 | 77 | 0.484
16 | 12 | 0 | 63 | 1.000
17 | 405 | 407 | 156 | 0.277
18 | 1 | 0 | 156 | 1.000
2 | 15 | 1 | 152 | 0.993
3 | 129 | 169 | 296 | 0.637
4 | 123 | 486 | 857 | 0.638
5 | 48 | 108 | 242 | 0.691
6 | 51 | 75 | 344 | 0.821
7 | 52 | 45 | 460 | 0.911
8 | 49 | 12 | 202 | 0.944
9 | 103 | 118 | 405 | 0.774
```
