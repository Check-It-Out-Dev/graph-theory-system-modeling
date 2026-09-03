# Subsystem 3 — dossier

size 129 (9.1%) · dominant Context 27% · ext-ratio 0.637 · flags: none

layers: {'Context': 35, 'Rule': 30, 'Resource': 29, 'Actor': 17, 'Process': 17, 'Event': 1}

terms: storage, yml, health, rate, geo, application, limit, upload, be2, ratelimit

medoids: InMemoryStorageRateLimitService.java, StorageRateLimitServiceUnitTest.java, StorageRateLimitService.java

entry: CucumberSpringConfig.java (44), RateLimitProfile.java (31), RateLimit.java (30), RateLimitKeyType.java (26), BaseServiceIntegrationTest.java (25)

actor roots: GeoIpAdminController.java, RateLimitCleanupTask.java, RateLimitPrivacyController.java, GeoLocationFacade.java, GeoLocationGdprController.java

seams: [{'other': 4, 'rel': 'IMPORTS', 'dir': 'in', 'n': 57}, {'other': 7, 'rel': 'IMPORTS', 'dir': 'out', 'n': 46}, {'other': 9, 'rel': 'IMPORTS', 'dir': 'in', 'n': 29}]

hyperedges (majority-in): 3, top: [{'metapath': 'A_P_A', 'hub': 'GdprCompliantRateLimiterService.java', 'arity': 3, 'idf': 2.862}, {'metapath': 'A_P_A', 'hub': 'TravelPatternService.java', 'arity': 3, 'idf': 2.862}, {'metapath': 'A_P_A', 'hub': 'StorageRateLimitService.java', 'arity': 4, 'idf': 2.457}]

folders: ['C:/Users/Norbert/IdeaProjects']
v3 overlap: {3: 122, 2: 1}
