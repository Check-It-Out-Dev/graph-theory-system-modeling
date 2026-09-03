# LEGACY (Neo4j era) — superseded by nav_delta_writer.py / nav_reclue_writer.py; kept for
# provenance; do not run against the Ladybug store. Its delete-and-recreate opening is
# RETIRED (ErdosNavigator ledger L3) and it still speaks bolt://.
#
# CodeMap Erdős E2+E3 — writes the navigation layer over the CURRENT (pre-curation) system:
# 19 SubsystemNavigator nodes (role CANDIDATE, provisional names) + CONTAINS_MEMBER edges,
# then 1 NavigationMaster with grouped index (fan-out <=9), mandated checklist, entry recipes.
# Prose in CLUES is authored, grounded ONLY in dossier fields + executed gold results.
# Numeric fields are COPIED from dossiers at runtime — never re-typed by hand.
#
# Usage: PYTHONUTF8=1 python nav_writer.py

import json, os
from neo4j import GraphDatabase

BOLT, AUTH, NS = "bolt://127.0.0.1:7611", ("neo4j", "password"), "CheckItOutV3"
DOSS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dossiers"))

# name / summary / responsibilities / caveats — grounded in dossier terms, medoids, seams,
# and the executed gold answers (mfq_all.jsonl). Provisional pending Grothendieck curation.
CLUES = {
    0: dict(name="User preferences & geo utilities",
            summary="Preference entities and rules plus geo/utility glue. Small, rule-heavy, highly external (0.939) — most of what it does is consumed elsewhere.",
            resp=["user preference entities + repositories", "geo lookup utilities", "shared util rules"],
            caveats=[]),
    1: dict(name="Address management",
            summary="Address entities, repositories and their integration tests. Feeds profile and company flows.",
            resp=["address entities + repos", "address service consumers", "integration test bases"],
            caveats=[]),
    2: dict(name="Social connections (fragment)",
            summary="The UserSocialConnection family. 1 internal edge vs 152 external (0.993) — structurally a fragment of the identity area, kept separate by clustering only.",
            resp=["social connection entity + repository", "social-auth linkage"],
            caveats=["MERGE CANDIDATE: external_ratio 0.993 — Grothendieck curation will judge"]),
    3: dict(name="Configuration & test context",
            summary="Spring profiles (application-*.yml), Cucumber/Spring test context, storage and health config. Lowest purity in the system (Context 26%) — a mixed platform bag.",
            resp=["profile ymls for every run mode", "Cucumber test wiring", "storage + health configuration"],
            caveats=["low purity (0.26) — expect heterogeneous members"]),
    4: dict(name="Partnership opportunities",
            summary="The campaign domain: PartnershipOpportunity and applied-opportunity entities, flows and permissions. Second-largest backend subsystem; entry PermissionUtils (73 external in-edges).",
            resp=["opportunity + applied-opportunity entities", "application/registration flows", "permission checks (PermissionUtils)"],
            caveats=["B-lens degeneracy possible: Resource-dominant"]),
    5: dict(name="Account deletion & Instagram",
            summary="Admin cascade deletion (preview/request/confirm family fanning over 21 files) plus the Instagram OAuth service. Deletion ORDER lives in AdminCascadeDeleteServiceImpl content.",
            resp=["cascade delete preview/execute", "Instagram OAuth + service", "deletion repositories"],
            caveats=["deletion sequence is file content — graph gives the touch set"]),
    6: dict(name="Step-up auth & user cache",
            summary="TOTP/step-up second factor and the Firestore-backed user cache. A service layer with no actor roots — invoked from authentication controllers, never self-starting.",
            resp=["TOTP + step-up flows", "UserCacheService / FirestoreService", "step-up token plumbing"],
            caveats=["UserCacheService single-carries the 11->6 seam (14/14 edges)"]),
    7: dict(name="Errors, logging & translatable messages",
            summary="Exception hierarchy, translatable error messages, logging configuration. ResourceNotFoundException alone takes 101 external in-edges — everything fails through here.",
            resp=["exception types + handlers", "translatable message keys", "logging config"],
            caveats=[]),
    8: dict(name="Validation & recaptcha rules",
            summary="Specification builders and recaptcha validation rules (entry SpecificationBuilder, 48 external in-edges). Rule-dominant (69%).",
            resp=["specification/query builders", "recaptcha verification", "unit-tested validation rules"],
            caveats=[]),
    9: dict(name="Authentication & registration",
            summary="Auth flows, registration, and the post-auth enforcement filters (banned-user, email-verification) that return 403 with a valid token. Third-largest subsystem.",
            resp=["registration + auth flows", "BannedUser/EmailVerification enforcement filters", "auth feature rules"],
            caveats=["the 403 answer lives here + ConsentEnforcementFilter in Billing (gold M04)"]),
    10: dict(name="User core & token exchange",
            summary="User.java itself (154 external in-edges — the entity everything imports), token exchange, email plumbing.",
            resp=["User entity + core repositories", "token exchange service", "email/token glue"],
            caveats=["LOAD-BEARING: User.java single-carries three seams (11->10 36/39, 12->10 18/18, 13->10 14/14)"]),
    11: dict(name="Billing, consent & legal",
            summary="The largest backend subsystem (208): Stripe payments, invoicing with retry, consent enforcement, legal documents, registry lookup, the payments boot guard and the lifecycle crons. Entry UserRepository.java (141 external in-edges).",
            resp=["Stripe subscription lifecycle + webhook", "invoicing (Fakturownia port) + retry cron", "consent capture/enforcement + GDPR crons", "legal documents + terms versioning", "campaign limits from plan"],
            caveats=["StorageUrlValidator (pentest 3.1 fix) post-dates the scan — coverage gap", "B-lens degeneracy possible: Resource-dominant"]),
    12: dict(name="Notifications & events",
            summary="Notification entities, transactional event listeners, email frequency preferences. The @TransactionalEventListener decoupling pattern lives here.",
            resp=["notification entities + listeners", "domain events (AccountActivatedEvent...)", "email frequency handling"],
            caveats=["TRIGGERS edges sparse system-wide (6) — event consumers under-modelled until delta enrichment"]),
    13: dict(name="Opportunity test suite (layer)",
            summary="90% Rule purity: the integration-test cluster for the opportunity domain (188-edge IMPORTS seam into Partnership opportunities). A LAYER, not a vertical slice.",
            resp=["integration test bases", "opportunity domain test coverage"],
            caveats=["RETYPE CANDIDATE role:'LAYER' — Grothendieck curation will judge"]),
    14: dict(name="FAQ & support content",
            summary="FAQ entities and categories backing the public support content.",
            resp=["FAQ entities + categories", "support content queries"],
            caveats=[]),
    15: dict(name="Support tickets",
            summary="Support ticket entities, attachments and flows — the public ticket create/status surface.",
            resp=["ticket entities + repos", "attachment handling", "ticket status flows"],
            caveats=["attachment URL validation (SSRF fix) is in the coverage gap set"]),
    16: dict(name="Account status fragment",
            summary="Twelve files around AccountStatus.java (51 external in-edges) with ZERO internal edges — a pure fragment held together by clustering, not structure.",
            resp=["AccountStatus + adjacent metadata"],
            caveats=["MERGE CANDIDATE: external_ratio 1.000, no internal edges — top curation item"]),
    17: dict(name="Greenfield frontend",
            summary="The entire Angular app (375 files): routes, standalone components, interceptor chain, i18n, SSR, sandbox fixtures and e2e tiers. No external in-edges — enter at the actor roots (app bootstrap), not via imports.",
            resp=["route table + guards", "feature components + layouts", "interceptor chain (6 indexed of 8)", "i18n (Transloco en/pl)", "sandbox fixtures + e2e"],
            caveats=["coverage gaps: set-to-array + ssr-cookie-forward + seo-title.strategy post-date the scan; e2e-tests partially indexed", "SPLIT CANDIDATE: MEGA flag, v3 overlap {0:180, 1:148} shows two latent halves"]),
    18: dict(name="Generated API client (collapsed)",
            summary="One node standing for the whole generated OpenAPI client (277 files collapsed by design). Carries all 150 FE->client IMPORTS — the articulation point between frontend and contract.",
            resp=["generated API services + models (regenerate via openapi:gen, never hand-edit)"],
            caveats=["deliberately collapsed; per-endpoint traceability is a possible enrichment"]),
}

GROUPS = [
    ("billing-legal", "payments, consent, user core, notifications", [11, 10, 12]),
    ("opportunities", "the campaign domain and its test layer", [4, 13]),
    ("auth-security", "authentication, second factor, validation rules", [9, 6, 8]),
    ("platform", "configuration, errors, shared utilities", [3, 7, 0]),
    ("frontend", "the Angular app and its generated API client", [17, 18]),
    ("data-support", "addresses, social links, deletion, FAQ, tickets, fragments", [1, 2, 5, 14, 15, 16]),
]

CHECKLIST = (
    "1. Read this node fully. 2. Try the answered-questions cache (alias match) - confident hit: "
    "answer and STOP. 3. Else match a Recipe by intent, fill slots, execute. 4. Else descend: "
    "subsystem_index -> matched L2 report -> entry_points/spines. 5. Open raw files only when "
    "clues are insufficient - and say so in the answer.")

SYSTEM_SUMMARY = (
    "checkItOut: a marketplace platform connecting companies with influencers (campaigns = "
    "PartnershipOpportunities; influencers apply). Spring Boot backend + Angular greenfield "
    "frontend, Stripe billing with Fakturownia invoicing, Firebase auth, GDPR consent "
    "enforcement, 10 lifecycle crons. 1374 indexed files in 19 candidate subsystems "
    "(pre-curation) across 6 navigation groups.")

GLOBAL_CAVEATS = [
    "scan is stale vs repo: 11 known coverage gaps (post-scan files, resources/, parts of e2e-tests/) - see mfq_all.jsonl COVERAGE_GAP records",
    "TRIGGERS (6) and TESTED_BY (104) edges are under-extracted - event-flow and test-coverage answers are shape signals",
    "subsystems are PRE-CURATION candidates: 16/2 merge, 13 retype, 17 split pending GrothendieckV5",
    "behavioural-lens embeddings degenerate on quiet Resources (research F88)",
]


def main():
    if "--i-mean-full-rebuild" not in __import__("sys").argv:
        raise SystemExit(
            "RETIRED (ErdosNavigator ledger L3): this first-run writer opens with DETACH DELETE "
            "of the nav layer and asserts 1374 members. The clue layer is now versioned with "
            "bi-temporal snapshots - use nav_delta_writer.py (supersession-safe). "
            "Override only for a from-scratch rebuild ordered by the owner: --i-mean-full-rebuild")
    doss = {}
    for f in os.listdir(DOSS):
        if f.startswith("subsystem_") and f.endswith(".json"):
            d = json.load(open(os.path.join(DOSS, f), encoding="utf-8"))
            doss[d["subsystem"]] = d

    drv = GraphDatabase.driver(BOLT, auth=AUTH)
    with drv.session() as s:
        s.run("MATCH (sn:SubsystemNavigator {namespace:$ns}) DETACH DELETE sn", ns=NS)  # idempotent rebuild of the nav layer only
        s.run("MATCH (nm:NavigationMaster {namespace:$ns}) DETACH DELETE nm", ns=NS)
        for sub, c in CLUES.items():
            d = doss[sub]
            s.run("""
                CREATE (sn:SubsystemNavigator {namespace:$ns, sub_id:$sub,
                  name:$name, role:'CANDIDATE', name_status:'PROVISIONAL',
                  ai_summary:$summary, responsibilities:$resp,
                  layer_profile:$lp, entry_points:$ep, spines:$spines,
                  contracts:$contracts, caveats:$caveats,
                  size:$size, external_ratio:$ext,
                  clue_version:'erdos-v1', generated_by:'ErdosNavigator/main-agent',
                  generated_at:datetime(), dossier_fingerprint:$dfp})
                WITH sn MATCH (m:V3Master {namespace:$ns}) CREATE (m)-[:GUIDES]->(sn)
                WITH sn MATCH (n:EntityDetail {namespace:$ns, v4_subsystem:$sub})
                CREATE (sn)-[:CONTAINS_MEMBER]->(n)""",
                  ns=NS, sub=sub, name=c["name"], summary=c["summary"], resp=c["resp"],
                  lp=json.dumps(d["layer_profile"]),
                  ep=json.dumps([e["name"] for e in d["entry_points"]] + d["actor_roots"]),
                  spines=json.dumps(d["top_hyperedges"]),
                  contracts=json.dumps(d["top_seams"]),
                  caveats=c["caveats"], size=d["size"], ext=d["external_ratio"],
                  dfp=__import__("hashlib").sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()[:16])
        idx_tree = "\n".join(
            f"{g}: {desc}\n" + "\n".join(
                f"  - [{sid}] {CLUES[sid]['name']} ({doss[sid]['size']} files) - "
                f"{CLUES[sid]['summary'].split('.')[0]}." for sid in subs)
            for g, desc, subs in GROUPS)
        s.run("""
            CREATE (nm:NavigationMaster {namespace:$ns, schema_version:'1.0',
              codebase_name:'checkItOut', scanned_at:'2026-09-02 scan (stale - see caveats)',
              ai_summary:$sys, ai_instruction:$chk, subsystem_index:$idx,
              global_caveats:$cav, clue_version:'erdos-v1',
              generated_by:'ErdosNavigator/main-agent', generated_at:datetime()})
            WITH nm MATCH (sn:SubsystemNavigator {namespace:$ns})
            CREATE (nm)-[:GUIDES]->(sn)""",
              ns=NS, sys=SYSTEM_SUMMARY, chk=CHECKLIST, idx=idx_tree, cav=GLOBAL_CAVEATS)
        chk = s.run("""
            MATCH (nm:NavigationMaster {namespace:$ns}) WITH count(nm) AS masters
            MATCH (sn:SubsystemNavigator {namespace:$ns}) WITH masters, count(sn) AS l2
            MATCH (:SubsystemNavigator {namespace:$ns})-[:CONTAINS_MEMBER]->(n)
            RETURN masters, l2, count(DISTINCT n) AS members""", ns=NS).single()
        print(f"VERIFIED BY READ: {chk['masters']} NavigationMaster, {chk['l2']} SubsystemNavigator, "
              f"{chk['members']} members under CONTAINS_MEMBER")
        assert (chk["masters"], chk["l2"], chk["members"]) == (1, 19, 1374)
    drv.close()
    print("fan-out check: 6 groups (<=9), max group size", max(len(g[2]) for g in GROUPS), "(<=9)")


if __name__ == "__main__":
    main()
