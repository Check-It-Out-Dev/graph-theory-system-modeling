# CodeMap Erdős E2+E3 RE-CLUE — bi-temporal supersession of the curated navigation layer.
#
# Successor to nav_delta_writer.py for the post-curation tree (owner decisions Q1=B, Q2=A).
# nav_writer.py's delete-and-recreate path stays RETIRED (ledger L3). Every node keeps its
# sub_id (foreign key for mfq depends_on_subsystems) and every outgoing body is preserved as
# an immutable :ClueSnapshot with (t_valid, t_invalid) plus a [:SUPERSEDED_BY] edge.
#
# Byte-identity discipline: prose is rewritten ONLY where a cited number moved, a claim became
# false, or there was no body at all. Subsystems in KEEP_PROSE re-stamp their structured fields
# from the refreshed dossier and keep their sentences byte-identical.
#
# STORE PORT (2026-09-02): Neo4j -> Ladybug Store. Bodies, FIXES, GROUPS and the L1 text
# are byte-identical to the executed 2026-09-02 batch; only the plumbing translated.
# Snapshots become ClueSnap rows (snap_id 'reclue-<batch>-<sub>', full pre-write state in
# `body`); typed Nav columns take direct SETs, long-tail fields ride props via merge_props;
# the L1 snapshot carries sub_id=-1 (the master has no sub) and level 'L1' in body.
#
# Usage: PYTHONUTF8=1 python nav_reclue_writer.py [--dry] [--l1]

import datetime as _dt
import hashlib, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "authoring")))
from ladybug_store import Store

DOSS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dossiers"))
CLUE_VERSION, BATCH = "curated-v2", "2026-09-02-reclue"
GENERATED_BY = "ErdosNavigator/reclue"
NOW = _dt.datetime.now().isoformat()

# Shared caveat: true for all nine frontend children, established from the hyperedge census
# (MATCH HyperedgeCandidate source:'metapath-v3' grouped by curated hub subsystem).
FE_NO_APR = ("no A_P_R spine exists in this child: every majority-internal hyperedge under it "
             "is A_P_A. Frontend Resources are models and fixtures, not repositories, so the "
             "Actor->Process->Resource walk that works on the backend has nothing to land on "
             "here - walk the A_P_A service hubs listed in spines instead")

# --- E2 bodies ---------------------------------------------------------------------------
# Every clause traces to a refreshed dossier field, a CurationDecision, or a named query.
CLUES = {
    # ---------- queue 1: the nine frontend children (clue_body_status was MISSING) ----------
    170: dict(
        summary="core/auth and feature/auth of the greenfield app: sign-in, sign-up and "
                "password-reset screens, the social callback, the step-up and two-factor "
                "client services, and the HTTP interceptor chain. Rule-dominant (36 of 69) "
                "because specs sit beside their subjects. Holds the largest internal seam in "
                "the whole app - core/auth to feature/auth, 40 edges - which is why the "
                "frontend was split vertically rather than by layer.",
        resp=["sign-in / sign-up / forgot-password screens and their specs",
              "auth API and session-state services (14 external in-edges each)",
              "step-up and two-factor client services",
              "the HTTP interceptor chain (set-to-array.interceptor.ts is the consumed one)",
              "social auth callback and the post-auth action router"],
        caveats=[FE_NO_APR,
                 "external_ratio 0.444 makes it the third most self-contained child, after "
                 "[178] at 0.176 and [172] at 0.0; cohesion 0.559"],
        evidence="dossier subsystem_170.json (size 69, layer_profile Rule 36/Actor 20/Process 7/"
                 "Resource 6, purity 0.522, external_ratio 0.444, entry_points, actor_roots, "
                 "top_seams, top_hyperedges, trophic_span [0.0, 2.0]); CurationDecision on "
                 "sub-17 for the 40-edge core/auth<->feature/auth seam and per-child cohesion "
                 "0.559; hyperedge census query for the A_P_A-only caveat"),
    171: dict(
        summary="Opportunity browsing and applied opportunities: list and detail screens, "
                "content submission, collaborations and grants, with their services and specs. "
                "Actor-dominant (24 of 43). Only one file is imported from outside the child "
                "(opportunity-dictionaries.service.ts, 1 external in-edge), so it is entered at "
                "its screens, not through an API.",
        resp=["applied-opportunity list and detail screens",
              "opportunity browsing and detail screens",
              "content submission flow",
              "collaborations and grants surfaces",
              "opportunity / applied-opportunity / content client services"],
        caveats=[FE_NO_APR,
                 "36 of its 43 files come from a single v3 cell (v3=12), so the child is a "
                 "cohort the previous partition already saw; cohesion 0.524"],
        evidence="dossier subsystem_171.json (size 43, layer_profile Actor 24/Rule 15/Process 4, "
                 "purity 0.558, external_ratio 0.476, entry_points 1, actor_roots, top_seams, "
                 "top_hyperedges A_P_A x3, trophic_span [0.0, 1.33]); CurationDecision on sub-17 "
                 "(OPPS 36/43 v3=12, cohesion 0.524)"),
    172: dict(
        summary="The onboarding survey and its showcase chapters (compliance, engineering, "
                "operations, platform, security and the per-topic showcases). The single most "
                "self-contained unit in the estate: cohesion 1.000, 14 internal edges and ZERO "
                "edges to any sibling or any other subsystem.",
        resp=["survey hub and chapter components",
              "per-topic showcase components",
              "survey chapter specs"],
        caveats=[FE_NO_APR + " - and this child has no majority-internal hyperedge at all "
                 "(hyperedges_majority 0), so it has no spine of any kind",
                 "external_ratio 0.0 is a measurement, not a gap: nothing imports into it and it "
                 "imports nothing, so an impact query starting anywhere else in the estate will "
                 "never reach it. Enter at survey-hub.component.ts",
                 "30 of its 42 members are flagged entry_point by E1 because every Actor with no "
                 "internal in-edge is an actor root here - read that as 'many independent "
                 "screens', not as 34 API surfaces"],
        evidence="dossier subsystem_172.json (size 42, layer_profile Actor 34/Rule 4/Resource 4, "
                 "purity 0.81, edges_internal 14, edges_external 0, external_ratio 0.0, "
                 "entry_points [], actor_roots, top_seams [], top_hyperedges [], trophic_span "
                 "[0.0, 2.25]); CurationDecision on sub-17 (SURVEY cohesion 1.000, 37/42 v3=1); "
                 "E1 read: 30 entry flags of 42 members"),
    173: dict(
        summary="A SUPPLIER LAYER, not a feature slice. It supplies the sandbox fixture data "
                "(sign-up, profile, opportunity-form and 20-odd more), the sandbox host / index "
                "/ registry that renders them, the icon audit, and the e2e harness including the "
                "integration _trace recorder and real-login. 85% Resource purity over 67 files "
                "with only 4 internal edges - it is consumed, it does not orchestrate.",
        resp=["sandbox fixture data for every feature child",
              "sandbox host, index and registry components",
              "e2e harness: integration _trace recorder, real-login, visual-parity pairs",
              "icon audit surface"],
        caveats=["cohesion 0.065 is EXPECTED and is the reason for the LAYER retype - a fixture "
                 "shelf has no internal story. Do not read it as a defect or queue a split",
                 "only 7 of its 67 members carry a local_height: the 4 internal edges touch 7 "
                 "nodes and the remaining 60 have no internal edge and no layer median to "
                 "inherit. Trophic questions about this child have no answer by construction",
                 "it INJECTS into the feature children (177 x37 IMPORTS, 170 x8, 174 x6, 176 x5) "
                 "rather than being called by them, so it will not appear on a caller->callee "
                 "walk that starts at a screen",
                 "real-login.ts sits here rather than in [170] by owner decision Q2=A (tier "
                 "coherence over a degree-0 cross-repo affinity to backend [9]); the dissent is "
                 "recorded on the node as curation_dissent"],
        evidence="dossier subsystem_173.json (size 67, layer_profile Resource 57/Rule 7/Actor 3, "
                 "purity 0.851, flag LAYER_PURITY, edges_internal 4, edges_external 58, "
                 "external_ratio 0.935, entry_points [], actor_roots 3, top_seams, medoids "
                 "sign-up/profile/opportunity-form fixtures); CurationDecision on sub-17 "
                 "(SANDBOX cohesion 0.065, retyped LAYER for that reason) and owner decision "
                 "Q2=A; E1 read: 7 of 67 with local_height, 3 entry flags"),
    174: dict(
        summary="Plan and billing screens with the subscription client, the frozen API models, "
                "and the consent / legal surface (reconsent dialog, upgrade and downgrade "
                "confirmations, legal API). Lowest purity of any frontend child (0.391) because "
                "screens, rules and services are all present in a 23-file space.",
        resp=["plan and billing screens",
              "upgrade / downgrade / reconsent dialogs",
              "subscription and public-config client services",
              "consent and legal API services",
              "frozen API models (hidden-models.ts)"],
        caveats=[FE_NO_APR,
                 "two of its files carry a measured cross-repo affinity to backend [11] "
                 "(subscription.client.ts and public-config.service.ts), confirmed in curation - "
                 "billing questions that cross the repo boundary land in [11]",
                 "cohesion 0.467"],
        evidence="dossier subsystem_174.json (size 23, layer_profile Actor 9/Rule 8/Process 5/"
                 "Resource 1, purity 0.391, external_ratio 0.533, entry_points hidden-models.ts "
                 "15 / public-config.service.ts 3 / consent.service.ts 2 / legal-api.service.ts "
                 "2, actor_roots, top_seams, top_hyperedges A_P_A subscription.service.ts, "
                 "trophic_span [0.0, 1.7]); CurationDecision on the 14 flagged nodes "
                 "(assignment_crossrepo_affinity 11 -> child 174, CONFIRMED) and on sub-17 "
                 "(BILLING cohesion 0.467)"),
    175: dict(
        summary="The support and help-centre screens: create-ticket, ticket status, the admin "
                "ticket list and detail, and the support-ticket client service that all of them "
                "share. Twenty files, all of them from one v3 cell.",
        resp=["create-ticket and ticket-status screens",
              "admin ticket list and detail screens",
              "support-ticket client service"],
        caveats=[FE_NO_APR,
                 "nothing outside the child imports into it (entry_points empty) - enter at "
                 "support.component.ts or the admin screens",
                 "no member carries a measured cross-repo affinity, so the pairing with backend "
                 "[15] is a name match, not a measurement; cohesion 0.529"],
        evidence="dossier subsystem_175.json (size 20, layer_profile Actor 12/Rule 7/Process 1, "
                 "purity 0.6, external_ratio 0.471, entry_points [], actor_roots, top_seams, "
                 "top_hyperedges A_P_A support-ticket.service.ts arity 6 idf 1.946, v3_overlap "
                 "{1: 20}); CurationDecision on sub-17 (SUPPORT 20/20 v3=1, cohesion 0.529) and "
                 "the crossrepo-affinity list, which names no file in this child"),
    176: dict(
        summary="Profile, user and settings surfaces plus the admin side: profile view and "
                "upload, preferences, security and social-connection settings, team, addresses, "
                "company, the admin user list and the dictionary editor. user.service.ts is the "
                "child's hub (11 external in-edges and the widest internal hyperedge).",
        resp=["profile view, upload and settings screens",
              "security, preferences and social-connection settings",
              "team, company and address surfaces",
              "admin user list and dictionary editor",
              "user / address / registry / cascade-delete client services"],
        caveats=[FE_NO_APR,
                 "two of its files carry a measured cross-repo affinity to backend [2], the "
                 "social-connection data model (social-connections.service.ts and "
                 "social-connections-settings.component.ts), confirmed in curation",
                 "cohesion 0.455, the lowest of the feature children - it is a settings drawer "
                 "of loosely related surfaces, so expect to enter at a named screen rather than "
                 "to walk it"],
        evidence="dossier subsystem_176.json (size 52, layer_profile Actor 26/Rule 18/Process 8, "
                 "purity 0.5, external_ratio 0.545, entry_points user.service.ts 11 / "
                 "registry.service.ts 2 / cascade-delete.service.ts 2 / address.service.ts 2, "
                 "actor_roots, top_seams, top_hyperedges A_P_A user.service.ts arity 7); "
                 "CurationDecision on the 14 flagged nodes (crossrepo 2 -> child 176, CONFIRMED) "
                 "and on sub-17 (PROFILE cohesion 0.455)"),
    177: dict(
        summary="The frontend's shared hub: layout and shell, theme, notification centre, i18n, "
                "shared utilities, health, error page, landing and marketing, the root build "
                "config and the bootstrap (main.ts / main.server.ts) - plus the whole generated "
                "OpenAPI client, collapsed into the single node `api` and merged here from "
                "ex-sub-18. Lowest purity in the estate (0.324) by design.",
        resp=["layout, shell and theme components",
              "notification centre, i18n (Transloco en/pl) and shared utilities",
              "landing, marketing, error page and health surfaces",
              "root build config and the SSR/browser bootstrap",
              "the collapsed generated OpenAPI client (`api`, 147 external in-edges)"],
        caveats=[FE_NO_APR,
                 "this is where every sibling's top seam points: 173 x37, 176 x33, 171 x29, "
                 "170 x20, 174 x14, 175 x12, 178 x2 IMPORTS inbound. If a frontend question is "
                 "about something shared, it is almost certainly here",
                 "the `api` node stands for 271 generated files collapsed by design - "
                 "per-endpoint traceability does not exist. Regenerate via openapi:gen, never "
                 "hand-edit. Of ex-sub-18's 156 in-edges, 147 remain external and 9 became "
                 "internal when it was merged into this child",
                 "rate-limit-state.service.ts carries a measured cross-repo affinity to backend "
                 "[3]; 8 of the 74 members have no local_height (no internal edge and no layer "
                 "median to inherit)"],
        evidence="dossier subsystem_177.json (size 74, layer_profile Actor 24/Rule 21/Resource 15/"
                 "Context 8/Process 6, purity 0.324, external_ratio 0.759, entry_points api 147 / "
                 "shell-status.service.ts 6 / rate-limit-state.service.ts 3 / number-format.ts 3, "
                 "actor_roots, top_seams all inbound); CurationDecision on sub-18 (external 156, "
                 "100% to sub-17, consumption by child SANDBOX 37 / PROFILE 33 / OPPS 29 / AUTH "
                 "20 / BILLING 14 / SUPPORT 12 / SHELL 9 / DEMO 2, max child share 23.7%); E1 "
                 "read: main.ts and main.server.ts are members, 66 of 74 with local_height"),
    178: dict(
        summary="Demo mode: the demo interceptor and demo-mode screens - demo hub and guide, the "
                "collab hero, and the Fakturownia / KSeF / inbox / phone-TOTP simulators driven "
                "by a scenario registry and sandbox director. Sixteen files, three external "
                "edges: external_ratio 0.176, the most self-contained child that touches "
                "anything at all.",
        resp=["demo hub, guide and collab hero screens",
              "Fakturownia / KSeF / inbox / phone-TOTP simulators",
              "scenario registry and demo fixtures",
              "the demo interceptor"],
        caveats=[FE_NO_APR,
                 "nothing imports into it (entry_points empty); its only outward edges are "
                 "IMPORTS x2 to [177] and INJECTS x1 to [174]. Enter at demo-hub.component.ts",
                 "cohesion 0.824, second only to the survey"],
        evidence="dossier subsystem_178.json (size 16, layer_profile Actor 8/Rule 4/Resource 3/"
                 "Process 1, purity 0.5, edges_internal 14, edges_external 3, external_ratio "
                 "0.176, entry_points [], actor_roots, top_seams, top_hyperedges A_P_A "
                 "sandbox-director.service.ts arity 6, medoids scenario-registry.ts / "
                 "demo-hub.component.ts / demo-fixtures.ts); CurationDecision on sub-17 "
                 "(DEMO cohesion 0.824, 14/16 v3=1)"),

    # ---------- queue 3 + 4: membership change and the LAYER reframes ----------
    4: dict(
        summary="The campaign domain: PartnershipOpportunity and applied-opportunity entities, "
                "their flows, permissions and reference data (dictionary, city, currency, "
                "service type, content type, platform) - and, since curation batch 2026-09-02, "
                "its own 49-file integration-test suite absorbed from ex-sub-13, because 60.3% "
                "of that suite's coupling pointed here. Second-largest subsystem after [11]. "
                "Entry PermissionUtils.java (68 external in-edges).",
        resp=["opportunity and applied-opportunity entities",
              "application / registration / active-cooperation flows",
              "permission checks (PermissionUtils, 68 external in-edges)",
              "reference-data services (dictionary, city, currency, service type, content type)",
              "the domain's 49-file integration-test suite (ex-sub-13, merged 2026-09-02)"],
        caveats=["the merge was measured, not assumed: cohesion 0.362 -> 0.478 (+32%), size 123 "
                 "-> 172 = 12.16% of the corpus, still under the 20% MEGA split trigger",
                 "DISSENT PRESERVED: the alternative reading of ex-sub-13 as a cross-cutting "
                 "LAYER was refuted (its direction profile was in=9 out=361, only 2 distinct "
                 "consumers - a consumer satellite, not a supplier) and is recorded on the "
                 "sub-13 node as curation_dissent. sub-13 is retained with role MERGED and a "
                 "[:SUPERSEDED_BY] edge, never deleted, so old mfq stamps naming 13 still resolve",
                 "B-lens degeneracy possible: Resource-dominant (90 of 172)",
                 "richest spine structure in the estate: 12 majority-internal A_P_R hyperedges "
                 "over 40 of its 172 members. Highest IDF is DictionaryService.java (arity 4, "
                 "idf 2.485); the widest is PartnershipOpportunityService.java (arity 10)"],
        evidence="dossier subsystem_4.json refreshed --curated (size 172, layer_profile Resource "
                 "90/Rule 49/Actor 17/Process 14, purity 0.523, edges_internal 718, "
                 "external_ratio 0.523, entry_points PermissionUtils.java 68 / BaseRepository."
                 "java 37 / DictionaryService.java 22, actor_roots, top_seams 7 x103 / 3 x57 / "
                 "1 x30, trophic_span [0.0, 2.69]); CurationDecision on sub-13 (60.3% coupling, "
                 "cohesion 0.362->0.478, 12.16%, dissent preserved); E1 A_P_R spine query"),
    2: dict(
        summary="A SUPPLIER LAYER: the UserSocialConnection and Platform data model with its "
                "repositories, DTOs, mapper and enum translation. 86.2% fan-in (in=138, out=22) "
                "across 8 distinct consumers. Merge was considered and REFUTED - external_ratio "
                "0.993 trips the first half of the merge trigger, but no partner dominates "
                "([4] 33.1% vs [5] 24.4%, a ratio of 1.36).",
        resp=["UserSocialConnection and Platform entities and repositories",
              "social-connection DTOs and mapper",
              "connection-status enum and its translation"],
        caveats=["the rejected merges were measured and both were negligible: into [4] cohesion "
                 "0.362 -> 0.372 (+0.010), into [5] 0.305 -> 0.311 (+0.006). Do not re-queue a "
                 "merge check on this node",
                 "corroborated across the repo boundary: the two flagged frontend "
                 "social-connection files carry assignment_crossrepo_affinity=2, i.e. the "
                 "unconstrained kNN independently named this subsystem - the behaviour of a "
                 "shared model, not of a fragment"],
        evidence="dossier subsystem_2.json (size 15, layer_profile Resource 11/Process 2/Context "
                 "1/Rule 1, purity 0.733, edges_internal 1, edges_external 152, external_ratio "
                 "0.993, entry_points UserSocialConnectionRepository.java 40 / "
                 "UserSocialConnection.java 30 / PlatformRepository.java 29 / ConnectionStatus."
                 "java 13, top_seams); CurationDecision on sub-2 (in=138 out=22, 86.2% fan-in, "
                 "8 consumers, partner shares, measured merge deltas, crossrepo corroboration)"),
    7: dict(
        summary="A SUPPLIER LAYER and the strongest one in the graph: the exception hierarchy, "
                "the translatable error-message infrastructure and the logging configuration "
                "that the whole backend imports. 91.1% fan-in (in=422, out=41) across 15 "
                "distinct consumers with no consumer above 13.4%. Everything fails through here.",
        resp=["exception types and handlers (BaseExceptionHandler, AuthenticationExceptionHandler)",
              "translatable message keys and the translatable-exception family",
              "logging and network-filter configuration"],
        caveats=["retyped to LAYER by curation without any membership change: it was not in the "
                 "queue, but the same fan-in criterion that promoted [2] and [16] selects this "
                 "node more strongly than either, and applying a criterion to some candidates "
                 "and not to others would make the taxonomy arbitrary",
                 "what it supplies is pure supply - ResourceNotFoundException 101 external "
                 "in-edges, ValidationTranslatableException 96, InsufficientPermissionsException "
                 "58, BusinessRuleTranslatableException 37, AuthenticationTranslatableException "
                 "37 - and it has no actor roots at all, so it never starts a flow",
                 "no majority-internal hyperedge, so no spine: it is a shelf, walk it by name"],
        evidence="dossier subsystem_7.json (size 52, layer_profile Resource 27/Rule 21/Context 3/"
                 "Actor 1, purity 0.519, edges_internal 45, edges_external 460, external_ratio "
                 "0.911, entry_points with the five in-degree counts, actor_roots [], "
                 "hyperedges_majority 0, top_seams 4 x103 / 3 x46 / 6 x46); CurationDecision on "
                 "sub-7 (in=422 out=41 fan-in share 91.1%, 15 consumers, max partner 13.4%)"),
    16: dict(
        summary="A SHARED-VOCABULARY LAYER: status and compensation enums, their i18n message "
                "bundles and the OpenAPI spec config. 93.7% fan-in across 12 distinct consumers "
                "with ZERO internal edges. Merge was considered and REFUTED - external_ratio "
                "1.000 trips the first half of the trigger but the top three partners are tied "
                "at 8 edges each (12.7%), which is a shared kernel, not a fragment of anybody.",
        resp=["AccountStatus and the status / compensation enum family (AccountStatus.java "
              "alone takes 51 external in-edges)",
              "status metadata and i18n message bundles",
              "OpenAPI spec configuration"],
        caveats=["every candidate merge LOWERED the host's cohesion - into [4] 0.362->0.353, "
                 "[6] 0.177->0.173, [9] 0.242->0.234, [11] 0.489->0.471. This was the top "
                 "curation item and it is now closed; do not re-queue it",
                 "zero internal edges means no trophic height and no spine: E1 leaves "
                 "local_height null on all 12 members BY CONSTRUCTION. That is correct for a "
                 "vocabulary shelf and is not missing data",
                 "the promotion contradicts the manual's purity>0.85 layer trigger (measured "
                 "purity is Resource 67%); the fan-in signature is the direct evidence and the "
                 "measurement was allowed to win"],
        evidence="dossier subsystem_16.json (size 12, layer_profile Resource 8/Context 3/Rule 1, "
                 "purity 0.667, edges_internal 0, edges_external 63, external_ratio 1.0, "
                 "entry_points AccountStatus.java 51, actor_roots [], trophic_span null); "
                 "CurationDecision on sub-16 (93.7% fan-in, 12 consumers, max partner 12.7%, "
                 "per-host merge cohesion deltas, purity-trigger contradiction); E1 read: 0 of "
                 "12 members carry a local_height"),
}

# --- minimal corrections: one clause each, everything else byte-identical -----------------
# Same principle nav_delta_writer used for sub-11: the grounding law ("the runtime model cannot
# detect drift between clue and graph, so drift must not exist") outranks byte-identity, which
# exists to forbid gratuitous rewrites, not necessary corrections.
FIXES = {
    0: dict(field="ai_summary",
            old="highly external (0.939)", new="highly external (0.94)",
            reason="cited external_ratio moved with the refreshed dossier (0.939 -> 0.94)"),
    1: dict(field="ai_summary",
            old="Feeds profile and company flows.",
            new="Its heaviest coupling is with the campaign domain [4]: IMPORTS 30 out and 28 in.",
            reason="'profile and company flows' traced to no dossier field or query; replaced "
                   "with the measured top seam from subsystem_1.json"),
    5: dict(field="ai_summary",
            old="(preview/request/confirm family fanning over 21 files)",
            new="(AdminCascadeDeleteServiceImpl.java fans out to 17 files; 26 of the 48 members "
                "are named for cascade or deletion)",
            reason="the 21-file claim does not reproduce against the post-delta graph; measured "
                   "out-degree is 17 distinct targets and in-degree 0"),
    6: dict(field="ai_summary",
            old="A service layer with no actor roots — invoked from authentication "
                "controllers, never self-starting.",
            new="Almost no actor roots — InMemoryUserCache.java is the only one; the rest is "
                "invoked from authentication controllers, never self-starting.",
            reason="'no actor roots' is false against the refreshed dossier, which lists "
                   "InMemoryUserCache.java"),
    8: dict(field="caveats",
            old="MERGE CHECK: external_ratio 0.944 (12 internal vs 202 external edges) — a "
                "rule shelf consumed elsewhere rather than a self-contained module, and the +4 "
                "validators this batch did not raise internal cohesion. Grothendieck curation "
                "should judge merge vs keep.",
            new="MERGE CHECK RESOLVED as KEEP (GrothendieckV5, 2026-09-02). external_ratio 0.944 "
                "trips the first half of the merge trigger, but no partner dominates ([4] 26.8%) "
                "and the direction profile is balanced (in=104 out=105, 12 consumers), so it is "
                "neither a fragment of one neighbour nor a supplier layer. The measured merge "
                "into [4] gained +0.006 cohesion. Do not re-queue.",
            reason="the caveat asked for a curation judgement that has since been made; leaving "
                   "it would make a reader queue resolved work"),
    9: dict(field="ai_summary",
            old="Third-largest subsystem.",
            new="Fourth-largest leaf subsystem, after [11] 208, [4] 172 and [3] 129.",
            reason="the ranking moved when [4] absorbed sub-13 (123 -> 172) and [3] grew to 129"),
    10: dict(field="caveats",
             old="LOAD-BEARING: User.java single-carries three seams (11->10 36/39, 12->10 18/18, "
                 "13->10 14/14)",
             new="LOAD-BEARING: User.java single-carries three seams (11->10 36/39, 12->10 18/18, "
                 "4->10 34/39). The third was 13->10 before sub-13 merged into [4]",
             reason="the third seam named sub-13, which no longer holds members; re-measured "
                    "against the curated membership"),
}

# Prose stays byte-identical; only structured fields and provenance are re-stamped.
KEEP_PROSE = [3, 11, 12, 14, 15]

# --- group navigators ---------------------------------------------------------------------
# A group answers exactly one question: which child do I enter? It never repeats child detail.
GROUPS = {
    17: dict(
        summary="The whole Angular greenfield app: 406 files, 100% frontend-repo pure, cohesion "
                "0.723 - the highest of any subsystem. Answer through a child, never through "
                "this node. The split is vertical by domain, chosen over a horizontal layer "
                "split on measured modularity (Q=0.604 vs 0.120, 5.0x). One routing fact settles "
                "most descents: every child with external edges sends its top seam to [177], so "
                "anything shared - shell, i18n, the generated API client - is [177].",
        children=[
            (177, "shell, i18n, theme, notifications, landing, bootstrap and the collapsed "
                  "generated OpenAPI client - the hub every sibling imports"),
            (170, "auth: sign-in/up, password reset, step-up, two-factor and the interceptor chain"),
            (173, "LAYER - sandbox fixtures and the e2e harness; supplies the others, orchestrates nothing"),
            (176, "profile, settings, addresses, team, company, admin user list and dictionary"),
            (171, "opportunity browsing, applied opportunities, content submission, grants"),
            (172, "the onboarding survey and its showcase chapters - zero edges to any sibling"),
            (174, "plan and billing screens, subscription client, consent and legal"),
            (175, "support tickets and the help centre, including the admin ticket screens"),
            (178, "demo mode: the demo interceptor, demo screens and the simulators"),
        ],
        caveats=["e2e and contract-test answers are shape signals at best: 125 of 132 "
                 "e2e-tests/*.ts are unindexed (bdd 0/24, integration 5/50, _framework 1/35, "
                 "visual-parity 1/5), plus 27 files under src/testing and 2 under src/mocks",
                 "seven members carry a measured cross-repo affinity naming a backend "
                 "counterpart: subscription.client.ts and public-config.service.ts -> [11] (in "
                 "child 174), social-connections.service.ts and "
                 "social-connections-settings.component.ts -> [2] (child 176), "
                 "rate-limit-state.service.ts -> [3] (child 177), recorder.ts and real-login.ts "
                 "-> [7] and [9] (child 173). Five of the seven landed in the child whose "
                 "backend counterpart the unconstrained kNN named - an independent "
                 "cross-validation of the split",
                 "no child has an A_P_R spine; frontend hyperedges are all A_P_A service hubs"],
        evidence="CurationDecision on sub-17 (modularity vertical Q=0.604 vs horizontal 0.120, "
                 "aggregate cohesion 0.489 vs 0.241, largest child share 5.16%, fan-out 9, "
                 "per-child cohesion list, Louvain Q=0.815 only at 182 communities); child "
                 "dossiers 170-178 for sizes and top_seams; CurationDecision on the 14 flagged "
                 "nodes for the seven crossrepo affinities"),
    201: dict(
        summary="283 backend files behind money, the user record and outbound messages. Enter "
                "[11] for anything about money, subscriptions, invoicing, consent or the "
                "lifecycle crons; [10] for who the user IS; [12] for telling them about it.",
        children=[
            (11, "Stripe payments, invoicing with retry, consent enforcement, legal documents, "
                 "registry lookup, the payments boot guard and the lifecycle crons"),
            (10, "the User entity itself (154 external in-edges, the most-imported node in the "
                 "estate), token exchange, Firebase and SMTP glue"),
            (12, "notification entities, @TransactionalEventListener domain events, email "
                 "frequency preferences"),
        ],
        caveats=["User.java in [10] is the articulation point of this group: it single-carries "
                 "the 11->10 seam (36 of 39 edges) and the 12->10 seam (18 of 18)"],
        evidence="dossiers subsystem_11/10/12.json (sizes 208/39/36, entry_points User.java 154 "
                 "and UserRepository.java 141); seam-carrier query over curated CONTAINS_MEMBER "
                 "membership for 11->10 and 12->10; c2b_groups.py owner decision Q1=B"),
    202: dict(
        summary="235 files of configuration and cross-cutting supply. Three of the five children "
                "are LAYERs - they are consumed, they do not orchestrate. Enter [7] for an error "
                "type or message key, [16] for a status enum or the OpenAPI contract, [2] for "
                "the social-connection model, [3] for anything with a yml, a rate limit or the "
                "Spring/Cucumber test context, [0] for preferences or geo distance.",
        children=[
            (3, "Spring profiles including dev-lite, the rate-limit family, storage and upload, "
                "health, geo-IP GDPR, and the Cucumber/Spring test context"),
            (7, "LAYER - the exception hierarchy and translatable messages the whole backend "
                "imports; 91.1% fan-in across 15 consumers"),
            (0, "user preference entities and rules, geo distance calculation, PII masking"),
            (2, "LAYER - the UserSocialConnection and Platform data model; 86.2% fan-in across "
                "8 consumers"),
            (16, "LAYER - status and compensation enums, i18n bundles and the OpenAPI spec "
                 "config; 93.7% fan-in across 12 consumers, zero internal edges"),
        ],
        caveats=["[3] has the lowest purity in the system (Context 27%) - it is a mixed platform "
                 "bag, so expect heterogeneous members rather than one story",
                 "the three LAYERs have no actor roots and never start a flow; a caller->callee "
                 "walk will only ever arrive at them"],
        evidence="dossiers subsystem_3/7/0/2/16.json (sizes 129/52/27/15/12, purities, "
                 "external_ratios); CurationDecisions on sub-7, sub-2 and sub-16 for the fan-in "
                 "shares and consumer counts; c2b_groups.py owner decision Q1=B"),
    203: dict(
        summary="203 files that decide whether a request is allowed to proceed. Enter [9] for "
                "how a user logs in, registers, or is blocked after login; [6] for the second "
                "factor and the cached user identity; [8] for a validation annotation, a query "
                "specification or a crypto helper.",
        children=[
            (9, "auth flows, registration, the post-auth enforcement filters that return 403 "
                "with a valid token, and the Cucumber scenario/actor harness"),
            (6, "TOTP and step-up second factor plus the Firestore-backed user cache "
                "(UserCacheService, 51 external in-edges)"),
            (8, "SpecificationBuilder (48 external in-edges), HMAC/TOTP crypto, validation "
                "annotations, recaptcha, Vimeo and social-post URL rules"),
        ],
        caveats=["the 403-with-a-valid-token answer spans this group and [11]: the enforcement "
                 "filters are in [9] but ConsentEnforcementFilter is in billing"],
        evidence="dossiers subsystem_9/6/8.json (sizes 103/51/49, entry_points UserCacheService."
                 "java 51 and SpecificationBuilder.java 48); CurationDecision on sub-8 (KEEP, "
                 "in=104 out=105, 12 consumers); c2b_groups.py owner decision Q1=B"),
    204: dict(
        summary="116 files of user-owned data and the surfaces that manage it. Enter [5] for "
                "removing a user and everything they own; [15] for a ticket or an attachment; "
                "[1] for an address; [14] for FAQ content.",
        children=[
            (5, "admin cascade deletion plus the Instagram OAuth and data-deletion callback "
                "services"),
            (15, "support ticket entities, attachments and flows, the public create/status "
                 "surface and ticket access tokens"),
            (1, "address entities, repositories, primary/copy resolution and their integration "
                "tests"),
            (14, "FAQ entities and categories backing the public support content"),
        ],
        caveats=["attachment answers span this group and [3]: StorageUrlValidator.java validates "
                 "attachment URLs but sits in the config bag [3], not in [15]",
                 "the cascade deletion ORDER is file content, not graph structure - the graph "
                 "gives the touch set, the sequence is inside AdminCascadeDeleteServiceImpl.java"],
        evidence="dossiers subsystem_5/15/1/14.json (sizes 48/35/18/15); the sub-15 and sub-3 "
                 "clue bodies for the StorageUrlValidator split; c2b_groups.py owner decision Q1=B"),
}


def fingerprint(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def struct_fields(d):
    """The structured, dossier-copied fields. Never re-derived here — copied verbatim."""
    return dict(
        lp=json.dumps(d["layer_profile"]),
        ep=json.dumps([f"{e['name']} ({e['ext_in']} ext in-edges)" for e in d["entry_points"]]
                      + [f"{n} (actor root)" for n in d["actor_roots"]]),
        spines=json.dumps(d["top_hyperedges"]),
        contracts=json.dumps(d["top_seams"]),
        size=d["size"], ext=d["external_ratio"], dfp=fingerprint(d))


NAV_COLS = {"name", "role", "routable", "parent", "size", "external_ratio", "ai_summary",
            "clue_version", "clue_body_status", "generated_by", "clue_delta_batch",
            "dossier_fingerprint", "spines", "entry_points", "contracts", "caveats",
            "responsibilities"}


def pull_nav(s, sub):
    """Full navigator state (typed cols + parsed props) as one flat dict."""
    r = s.one("MATCH (n:Nav) WHERE n.sub_id = $sub RETURN "
              + ", ".join(f"n.{c} AS {c}" for c in sorted(NAV_COLS))
              + ", n.props AS props", dict(sub=sub))
    if r is None:
        return None
    p = json.loads(r.pop("props") or "{}")
    return {**p, **r}


def set_nav(s, sub, props):
    """SET on a Nav row: typed columns directly (lists jdumped; JSON-looking strings as
    armored literals; None as literal NULL), everything else into props via merge_props."""
    cols, extra = {}, {}
    for k, v in props.items():
        (cols if k in NAV_COLS else extra)[k] = v
    sets, params = [], {"sub": sub}
    for k, v in sorted(cols.items()):
        if isinstance(v, list):
            v = json.dumps(v, ensure_ascii=False)
        if v is None:
            sets.append(f"n.{k} = NULL")
        elif isinstance(v, str) and v.lstrip()[:1] in ("[", "{"):
            sets.append(f"n.{k} = {Store.lit(v)}")
        else:
            sets.append(f"n.{k} = ${k}")
            params[k] = v
    if sets:
        s.conn.execute("MATCH (n:Nav) WHERE n.sub_id = $sub SET " + ", ".join(sets),
                       parameters=params)
    if extra:
        s.merge_props("Nav", "sub_id", sub, extra)


def snapshot_and_set(s, sub, props, reason, has_body):
    """Bi-temporal supersession. The snapshot is OFF-TRAVERSAL: a ClueSnap row holding the
    full pre-write state in `body`; supersession is recorded in its columns, never an edge."""
    if has_body:
        body = dict(pull_nav(s, sub), level="L2", reason=reason)
        s.create("ClueSnap", dict(
            snap_id=f"reclue-{BATCH}-{sub}", sub_id=sub, taken_at=NOW, t_created=NOW,
            t_expired="", superseded_by=BATCH,
            body=json.dumps(body, ensure_ascii=False, default=str)), "snap_id")
    set_nav(s, sub, dict(props, clue_version=CLUE_VERSION, generated_by=GENERATED_BY,
                         clue_delta_batch=BATCH, clue_body_status="CURRENT",
                         generated_at=NOW))


def write_l2(s, doss, dry):
    written, kept = [], []
    for sub, c in CLUES.items():
        sf = struct_fields(doss[sub])
        props = dict(ai_summary=c["summary"], responsibilities=c["resp"], caveats=c["caveats"],
                     clue_evidence=c["evidence"], layer_profile=sf["lp"],
                     entry_points=sf["ep"], spines=sf["spines"], contracts=sf["contracts"],
                     size=sf["size"], external_ratio=sf["ext"], dossier_fingerprint=sf["dfp"])
        if dry:
            print(f"DRY sub-{sub}: full body, size={sf['size']} ext={sf['ext']} fp={sf['dfp']}")
            continue
        has = s.one("MATCH (n:Nav) WHERE n.sub_id = $sub "
                    "RETURN n.ai_summary IS NOT NULL AS h", dict(sub=sub))["h"]
        snapshot_and_set(s, sub, props, f"re-clue batch {BATCH}: full body rewritten for the "
                         f"curated tree", has)
        written.append(sub)

    for sub, fix in FIXES.items():
        sf = struct_fields(doss[sub])
        props = dict(layer_profile=sf["lp"], entry_points=sf["ep"], spines=sf["spines"],
                     contracts=sf["contracts"], size=sf["size"], external_ratio=sf["ext"],
                     dossier_fingerprint=sf["dfp"], clue_correction_note=fix["reason"])
        if dry:
            print(f"DRY sub-{sub}: one-clause fix on {fix['field']}")
            continue
        cur = s.one("MATCH (n:Nav) WHERE n.sub_id = $sub "
                    "RETURN n.ai_summary AS s, n.caveats AS c", dict(sub=sub))
        if fix["field"] == "ai_summary":
            assert cur["s"].count(fix["old"]) == 1, f"sub-{sub} anchor not unique: {fix['old'][:50]!r}"
            props["ai_summary"] = cur["s"].replace(fix["old"], fix["new"])
        else:
            cavs = json.loads(cur["c"] or "[]")   # caveats column is a JSON string in the store
            assert cavs.count(fix["old"]) == 1, f"sub-{sub} caveat anchor not found"
            props["caveats"] = [fix["new"] if x == fix["old"] else x for x in cavs]
        snapshot_and_set(s, sub, props, f"re-clue batch {BATCH}: {fix['reason']}", True)
        written.append(sub)

    for sub in KEEP_PROSE:
        sf = struct_fields(doss[sub])
        props = dict(layer_profile=sf["lp"], entry_points=sf["ep"], spines=sf["spines"],
                     contracts=sf["contracts"], size=sf["size"], external_ratio=sf["ext"],
                     dossier_fingerprint=sf["dfp"])
        if dry:
            print(f"DRY sub-{sub}: prose byte-identical, structured fields re-stamped")
            continue
        snapshot_and_set(s, sub, props, f"re-clue batch {BATCH}: structured fields re-stamped "
                         f"from the refreshed dossier; prose byte-identical", True)
        kept.append(sub)
    return written, kept


def write_groups(s, doss, dry):
    out = []
    for gid, g in GROUPS.items():
        kids = [(k, one) for k, one in g["children"]]
        sizes = {str(k): doss[k]["size"] for k, _ in kids}
        total = sum(sizes.values())
        index = "\n".join(f"  - [{k}] {doss[k]['size']} files - {one}" for k, one in kids)
        agg = {}
        for k, _ in kids:
            for layer, n in doss[k]["layer_profile"].items():
                agg[layer] = agg.get(layer, 0) + n
        # A group has no measured boundary of its own: external_ratio, spines and contracts are
        # properties of leaves. Setting them null REMOVES the stale leaf-era values (sub-17 still
        # carried numbers from its pre-curation 405-file dossier) rather than leaving drift.
        props = dict(ai_summary=g["summary"], caveats=g["caveats"], clue_evidence=g["evidence"],
                     child_index=index, child_sub_ids=[k for k, _ in kids],
                     child_sizes=json.dumps(sizes), size=total,
                     layer_profile=json.dumps(dict(sorted(agg.items(), key=lambda x: -x[1]))),
                     entry_points=json.dumps(["a group is not entered directly - route through "
                                              "child_index to a leaf navigator"]),
                     spines=None, contracts=None, external_ratio=None,
                     responsibilities=[f"route to [{k}]: {one}" for k, one in kids],
                     dossier_fingerprint=fingerprint(sizes))
        if dry:
            print(f"DRY group-{gid}: {len(kids)} children, {total} files, fp={props['dossier_fingerprint']}")
            continue
        row = s.one("MATCH (n:Nav) WHERE n.sub_id = $sub "
                    "RETURN n.ai_summary IS NOT NULL AS h", dict(sub=gid))
        if row is None:   # group navigators 201-204 are created by c2b_groups.py first
            sys.exit(f"FATAL: group navigator {gid} missing — run c2b_groups.py before re-clue")
        snapshot_and_set(s, gid, props, f"re-clue batch {BATCH}: group body written "
                         f"(which-child-do-I-enter)", row["h"])
        out.append(gid)
    return out


# --- E3 / L1 -------------------------------------------------------------------------------
L1_SUMMARY = (
    "checkItOut: a marketplace platform connecting companies with influencers (campaigns = "
    "PartnershipOpportunities; influencers apply). Spring Boot backend + Angular greenfield "
    "frontend, Stripe billing with Fakturownia invoicing, Firebase auth, GDPR consent "
    "enforcement, 10 lifecycle crons. 1415 indexed files in a curated 6-way tree: 4 group "
    "navigators, the frontend group with 9 children, and the campaign domain directly under "
    "this node. 17 leaf backend navigators (13 slices, 3 layers) plus the 9 frontend children.")

L1_INDEX = """- [201] GROUP Billing, identity & notifications (283 files) - money, the user record, outbound messages.
  - [11] Subscriptions, payments & consent (208 files) - Stripe payments, invoicing with retry, consent enforcement, legal documents, registry lookup, the payments boot guard and the lifecycle crons. Entry UserRepository.java (141 external in-edges).
  - [10] User identity & token exchange (39 files) - the User entity (154 external in-edges, the most-imported node in the estate), token exchange, Firebase and SMTP glue.
  - [12] Notifications & domain events (36 files) - notification entities, @TransactionalEventListener domain events, email frequency preferences.
- [202] GROUP Platform runtime & shared layers (235 files) - configuration and cross-cutting supply; 3 of the 5 children are LAYERs.
  - [3] Rate limits & runtime config (129 files) - Spring profiles (application-*.yml, incl. dev-lite), the rate-limit family, storage and upload, health, geo-IP GDPR, Cucumber/Spring test context. Lowest purity in the system (Context 27%).
  - [7] LAYER Translatable exceptions & logging (52 files) - the exception hierarchy and translatable messages the whole backend imports: 91.1% fan-in across 15 consumers, ResourceNotFoundException alone at 101 external in-edges.
  - [0] User preferences & geo distance (27 files) - preference entities and rules, geo distance calculation, PII masking utilities.
  - [2] LAYER Social connection data model (15 files) - UserSocialConnection and Platform entities, repositories, DTOs and mapper: 86.2% fan-in across 8 consumers.
  - [16] LAYER Status enums & OpenAPI contract (12 files) - status and compensation enums, i18n message bundles and the OpenAPI spec config: 93.7% fan-in across 12 consumers, zero internal edges.
- [203] GROUP Authentication & validation (203 files) - whether a request is allowed to proceed.
  - [9] Auth journeys & BDD harness (103 files) - auth flows, registration, the post-auth enforcement filters that return 403 with a valid token, and the Cucumber scenario/actor harness.
  - [6] Two-factor auth & user cache (51 files) - TOTP and step-up second factor plus the Firestore-backed user cache (UserCacheService, 51 external in-edges).
  - [8] Validation, crypto & query specs (49 files) - SpecificationBuilder (48 external in-edges), HMAC/TOTP crypto, validation annotations, recaptcha, Vimeo and social-post URL rules.
- [204] GROUP Data, deletion & support (116 files) - user-owned data and the surfaces that manage it.
  - [5] Account deletion & Instagram sync (48 files) - admin cascade deletion (AdminCascadeDeleteServiceImpl.java fans out to 17 files) plus the Instagram OAuth and data-deletion callback services.
  - [15] Support tickets & attachments (35 files) - ticket entities, attachments and flows, the public create/status surface, plus TicketAccessTokenService.
  - [1] Address resolution & storage (18 files) - address entities, repositories, primary/copy resolution and their integration tests.
  - [14] FAQ content & categories (15 files) - FAQ entities and categories backing the public support content.
- [17] GROUP Greenfield Angular frontend (406 files) - the whole Angular app, 100% FE-repo pure, cohesion 0.723 (the highest of any subsystem). Answer through a child, never through the group; anything shared is [177].
  - [177] FE shell, i18n & generated client (74 files) - layout, shell, theme, notifications, i18n, shared, health, error page, landing, root build config, the bootstrap, and the collapsed generated OpenAPI client (`api`, 147 external in-edges). Every sibling's top seam points here.
  - [170] FE auth, 2FA & interceptors (69 files) - sign-in/up, password reset, step-up, two-factor and the interceptor chain. Holds the app's largest internal seam: core/auth to feature/auth, 40 edges.
  - [173] LAYER FE fixtures & E2E harness (67 files) - sandbox fixtures plus the e2e harness. 85% Resource purity, 4 internal edges, cohesion 0.065 by design: it supplies, it does not orchestrate.
  - [176] FE profile, settings & admin (52 files) - profile, upload, settings, social connections, preferences, addresses, company, team, admin user list and dictionary.
  - [171] FE opportunities & applications (43 files) - opportunity browsing, applied opportunities, content submission, collaborations and grants.
  - [172] FE onboarding survey (42 files) - the onboarding survey and its showcase chapters. Fully self-contained: cohesion 1.000, zero edges to any sibling.
  - [174] FE plan, billing & consent (23 files) - plan and billing screens, subscription client, frozen API models, consent and legal. Two files carry a measured cross-repo affinity to [11].
  - [175] FE support & help centre (20 files) - support ticket and help-centre screens with their client service.
  - [178] FE demo mode (16 files) - the demo interceptor, demo screens and the Fakturownia/KSeF/inbox/TOTP simulators. Cohesion 0.824.
- [4] Partnership opportunity lifecycle (172 files) - the campaign domain: PartnershipOpportunity and applied-opportunity entities, flows, permissions and reference data, now including its own 49-file integration-test suite (ex-sub-13, merged 2026-09-02: 60.3% of that suite's coupling pointed here). Entry PermissionUtils.java (68 external in-edges)."""

GLOBAL_CAVEATS_V3 = [
    "coverage gaps: 9 of the 11 documented COVERAGE_GAP records are closed by delta batch "
    "2026-09-02 - verified by re-running each record's own locator against the graph, not "
    "assumed (BE37, FE09, FE11, FE18, FE19, FE22, FE23, FE26, FE33). The remaining 2 are NOT "
    "backlog items: BE29 is a PERMANENT REFUSAL - the 4 credential-bearing files in BE "
    "src/main/resources (keystore.p12, service-account.json, service-accountProd.json, "
    "dashboard-config.json) are deliberately never indexed, because indexing them would send "
    "secrets to the remote embedding service; FE10 needs no new file, since ApiConfiguration "
    "lives inside the collapsed generated-client node (now child 177). Do not queue either as work",
    "remainder, measured post-batch and reproducible: 270 files are genuinely unindexed = 61 "
    "in-scope + 125 of 132 e2e-tests/*.ts + 84 of 104 BE src/main/resources. e2e-tests (bdd "
    "0/24, integration 5/50, _framework 1/35, visual-parity 1/5) and resources/ were never in "
    "the scan scope, so absence there is scope, not failure. The 271 generated-client files "
    "are collapsed into one node by design, not missing. SEPARATELY, and do not conflate the "
    "two: 321 already-indexed files are fingerprint-stale (318 content-changed + 3 mtime-only) "
    "- their nodes, edges and embeddings exist, only the source moved since the scan",
    "TRIGGERS (6) and TESTED_BY (113) edges are under-extracted - event-flow and test-coverage "
    "answers are shape signals",
    "subsystems are CURATED (GrothendieckV5 batch 2026-09-02-curation) and this index is a 6-way "
    "tree (owner decision Q1=B): NavigationMaster -> 4 GROUP navigators (201-204) + the frontend "
    "GROUP [17] + [4] directly, then 17 leaf backend navigators (13 slices, 3 layers) and the "
    "frontend's 9 children (170-178). Fan-out is <=9 at every level and all 1415 members sit "
    "<=3 hops from this node (measured: 578 reachable at hop 2, 1243 at hop 3, union 1415). "
    "Layers were promoted on a measured fan-in criterion applied uniformly to all 19 candidates "
    "(in-share >= 85%, >= 8 distinct consumers, no seam partner above 40%); it selects exactly "
    "2, 7 and 16. sub-13 merged into [4] and sub-18 into child [177]; both navigators are "
    "RETAINED with role MERGED and a [:SUPERSEDED_BY] edge to their absorber, never deleted, so "
    "cached answers stamped with 13 or 18 still resolve. Curated membership lives in "
    "CONTAINS_MEMBER plus curated_subsystem on the 455 moved or split nodes; v4_subsystem is the "
    "untouched measured layer, so the two disagree BY DESIGN - read CONTAINS_MEMBER for "
    "navigation and v4_subsystem only for provenance",
    "behavioural-lens embeddings degenerate on quiet Resources (research F88)",
    "clue layer is curated-v2 (re-clue batch 2026-09-02-reclue): all 30 live navigators now carry "
    "a written body with clue_body_status='CURRENT'. The 9 frontend children and the 4 GROUP "
    "navigators received their first body; 17 nodes were superseded in place, each with an "
    "off-traversal :ClueSnapshot and a [:SUPERSEDED_BY] edge; and 5 (3, 11, 12, 14, 15) kept "
    "their prose byte-identical because no number they cite moved. The 2 MERGED navigators (13, "
    "18) keep their final pre-merge body by design. Cached answers listed in "
    "eval/q/INVALIDATED_2026-09-02-curated.json must be dropped",
    "frontend spines are A_P_A only: no child of [17] has a majority-internal A_P_R hyperedge, "
    "because frontend Resources are models and fixtures rather than repositories. An "
    "Actor->Process->Resource walk works on the backend and returns nothing on the frontend",
]

ORG_REPAIR_NOTE = (
    "E1 org properties (layer, local_height, entry_point, spine_membership) were recomputed by "
    "c3_organisation.py on 2026-09-02 after a defect repair: in --curated mode the "
    "node->subsystem remap had been applied to nodes only, leaving every edge on v4_subsystem. "
    "Measured effect before the repair: int_edges for children 170-178 were empty, so 0 of 406 "
    "frontend members carried a local_height and entry_point had degenerated to every Actor "
    "(counts 20/24/34/3/9/12/26/24/8 = exactly each child's Actor count); sub-4's heights were "
    "computed on its pre-merge digraph (span 0-3.049 instead of the dossier's 0-2.69). After "
    "the repair E1's entry points reproduce the independently-computed dossiers exactly and "
    "sub-4's span matches at 2.69. The frozen-recipe conformance differential was re-run and "
    "stayed green (sub-11 service median 0.97 / 3 inversions, sub-4 0.89 / 3 inversions, zero "
    "drift).")


def write_l1(s, dry):
    m = s.one("MATCH (nm:Master) RETURN nm.mid AS mid, nm.ai_summary AS summ, "
              "nm.subsystem_index AS idx, nm.global_caveats AS cav, nm.props AS props")
    mp = json.loads(m["props"] or "{}")
    print(f"L1: index {len((m['idx'] or '').splitlines())} lines -> "
          f"{len(L1_INDEX.splitlines())} lines (6-way tree), caveats -> {len(GLOBAL_CAVEATS_V3)}")
    if dry:
        return
    body = dict(level="L1", clue_version=mp.get("clue_version"), ai_summary=m["summ"],
                subsystem_index=m["idx"], global_caveats=m["cav"],
                t_valid=mp.get("generated_at"),
                reason=f"re-clue batch {BATCH}: subsystem_index regenerated as the curated "
                       f"6-way tree; global caveats updated for written clue bodies")
    s.create("ClueSnap", dict(   # the master has no sub: sub_id=-1, level lives in body
        snap_id=f"reclue-{BATCH}-L1", sub_id=-1, taken_at=NOW, t_created=NOW,
        t_expired="", superseded_by=BATCH,
        body=json.dumps(body, ensure_ascii=False, default=str)), "snap_id")
    cav = json.dumps(GLOBAL_CAVEATS_V3, ensure_ascii=False)
    s.conn.execute(
        "MATCH (nm:Master) WHERE nm.mid = $mid SET nm.ai_summary = $summ, "
        f"nm.subsystem_index = $idx, nm.global_caveats = {Store.lit(cav)}",
        parameters=dict(mid=m["mid"], summ=L1_SUMMARY, idx=L1_INDEX))
    s.merge_props("Master", "mid", m["mid"], dict(
        superseded_summary_curation=m["summ"], superseded_index_reclue=m["idx"],
        clue_version=CLUE_VERSION, generated_by=GENERATED_BY, generated_at=NOW,
        clue_delta_batch=BATCH, clue_body_status="CURRENT", child_count=6,
        org_repair_note=ORG_REPAIR_NOTE))
    chk = s.one("MATCH (nm:Master) WHERE nm.mid = $mid RETURN nm.subsystem_index AS idx, "
                "nm.ai_summary AS summ, nm.global_caveats AS cav, nm.props AS props",
                dict(mid=m["mid"]))
    cp = json.loads(chk["props"] or "{}")
    assert cp.get("ai_instruction") == mp.get("ai_instruction"), \
        "ai_instruction (the mandated checklist) must stay verbatim"
    assert chk["idx"] == L1_INDEX and chk["summ"] == L1_SUMMARY, "L1 body did not land"
    assert json.loads(chk["cav"]) == GLOBAL_CAVEATS_V3, "global caveats did not land"
    print(f"VERIFIED BY READ: L1 clue_version={cp.get('clue_version')}, "
          f"{len(json.loads(chk['cav']))} global caveats, checklist verbatim=True")


def main():
    dry, l1_only = "--dry" in sys.argv, "--l1" in sys.argv
    doss = {}
    for f in os.listdir(DOSS):
        if f.startswith("subsystem_") and f.endswith(".json"):
            d = json.load(open(os.path.join(DOSS, f), encoding="utf-8"))
            doss[d["subsystem"]] = d

    s = Store(read_only=dry)
    if not l1_only:
        written, kept = write_l2(s, doss, dry)
        groups = write_groups(s, doss, dry)
        if not dry:
            print(f"L2: {len(written)} bodies written, {len(kept)} prose-identical, "
                  f"{len(groups)} group bodies")
    write_l1(s, dry)
    if dry:
        return
    rows = s.q("MATCH (sn:Nav) RETURN sn.sub_id AS sub, sn.role AS role, "
               "sn.clue_version AS cv, sn.clue_body_status AS st, sn.size AS size, "
               "sn.external_ratio AS ext, sn.dossier_fingerprint AS fp, "
               "sn.ai_summary AS summ")
    rows.sort(key=lambda r: r["sub"])            # determinism at the boundary
    snapc = {}
    for r in s.q("MATCH (c:ClueSnap) RETURN c.sub_id AS sid"):
        snapc[r["sid"]] = snapc.get(r["sid"], 0) + 1
    print("VERIFIED BY READ:")
    bad = []
    for r in rows:
        slen = len(r["summ"] or "")
        mark = "*" if r["cv"] == CLUE_VERSION else " "
        print(f" {mark} sub-{r['sub']:<3} {str(r['role']):<7} {str(r['cv']):<10} "
              f"{str(r['st']):<9} size={str(r['size']):<5} ext={str(r['ext']):<6} "
              f"fp={r['fp']} snaps={snapc.get(r['sub'], 0)} summary_chars={slen}")
        if r["role"] not in ("MERGED",):
            if r["cv"] != CLUE_VERSION or r["st"] != "CURRENT" or slen == 0:
                bad.append(r["sub"])
    assert not bad, f"live navigators left un-reclued: {bad}"
    for sub in list(CLUES) + list(FIXES) + KEEP_PROSE:
        r = next(x for x in rows if x["sub"] == sub)
        assert r["size"] == doss[sub]["size"], f"sub-{sub} size mismatch"
        assert r["fp"] == fingerprint(doss[sub]), f"sub-{sub} fingerprint mismatch"
    print(f"OK: {len(rows)} navigators, all live ones at {CLUE_VERSION}/CURRENT; "
          f"sizes and fingerprints match the refreshed dossiers")


if __name__ == "__main__":
    main()
