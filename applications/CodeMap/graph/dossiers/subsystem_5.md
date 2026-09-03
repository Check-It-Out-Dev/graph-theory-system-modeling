# Subsystem 5 — dossier

size 48 (3.4%) · dominant Resource 50% · ext-ratio 0.691 · flags: none

layers: {'Resource': 24, 'Rule': 12, 'Process': 7, 'Actor': 4, 'Context': 1}

terms: cascade, deletion, delete, orchestrator, instagram, be2, task, callback, data, test

medoids: AdminCascadeDeleteServiceImpl.java, InstagramDataDeletionService.java, AdminCascadeDeleteService.java

entry: InstagramService.java (8), UserAccountOrchestrator.java (6), DeletionEligibilityDto.java (5), OAuthCallbackService.java (3), HtmlEncoder.java (3)

actor roots: AdminCascadeDeleteController.java, OrphanCleanupTask.java, InstagramCallbackController.java

seams: [{'other': 4, 'rel': 'IMPORTS', 'dir': 'out', 'n': 28}, {'other': 10, 'rel': 'IMPORTS', 'dir': 'out', 'n': 23}, {'other': 2, 'rel': 'IMPORTS', 'dir': 'out', 'n': 23}]

hyperedges (majority-in): 1, top: [{'metapath': 'A_P_A', 'hub': 'AdminCascadeDeleteService.java', 'arity': 3, 'idf': 2.862}]

folders: ['C:/Users/Norbert/IdeaProjects']
v3 overlap: {14: 36, 2: 10}
