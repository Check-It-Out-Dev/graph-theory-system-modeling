# Subsystem 11 — dossier

size 208 (14.7%) · dominant Resource 53% · ext-ratio 0.509 · flags: none

layers: {'Resource': 110, 'Rule': 58, 'Actor': 20, 'Context': 9, 'Process': 9, 'Event': 2}

terms: subscription, consent, registry, adapter, legal, be2, stripe, instagram, test, payments

medoids: subscription-e2e.feature, consent-lifecycle.feature, SubscriptionPaidController.java

entry: UserRepository.java (141), UserType.java (48), LegalConsentService.java (18), CorsProperties.java (12), Permission.java (10)

actor roots: ConsentAdminController.java, ConsentController.java, LegalAdminController.java, LegalController.java, NoConsentAccountCleanupCronJob.java

seams: [{'other': 7, 'rel': 'IMPORTS', 'dir': 'out', 'n': 42}, {'other': 10, 'rel': 'IMPORTS', 'dir': 'out', 'n': 36}, {'other': 4, 'rel': 'IMPORTS', 'dir': 'out', 'n': 28}]

hyperedges (majority-in): 17, top: [{'metapath': 'A_P_R', 'hub': 'StripeService.java', 'arity': 3, 'idf': 2.89}, {'metapath': 'A_P_A', 'hub': 'ConsentService.java', 'arity': 3, 'idf': 2.862}, {'metapath': 'A_P_A', 'hub': 'LegalDocumentService.java', 'arity': 3, 'idf': 2.862}]

folders: ['C:/Users/Norbert/IdeaProjects']
v3 overlap: {6: 78, 7: 72}
