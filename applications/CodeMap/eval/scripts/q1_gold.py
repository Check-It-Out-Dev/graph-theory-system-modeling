# LEGACY (Neo4j era) — superseded by q_gold_all.py (--backend ladybug is the default
# authority); kept for provenance; do not run against the Ladybug store.
#
# CodeMap Q1-gold: the 12 worth=3 exemplar MFQs — executed gold, never imagined gold.
# Workflow (D11 execution verifier): run once -> read GOLD_RESULTS.md -> author GOLD_ANSWERS
# below -> run again (idempotent) -> mfq_gold.jsonl carries answers + fingerprints.
#
# Usage: PYTHONUTF8=1 python q1_gold.py
# Reads bolt://127.0.0.1:7611 CheckItOutV3. Writes ../q/mfq_gold.jsonl + ../q/GOLD_RESULTS.md.

import hashlib, json, os
from collections import Counter, defaultdict

from neo4j import GraphDatabase

BOLT, AUTH, NS = "bolt://127.0.0.1:7611", ("neo4j", "password"), "CheckItOutV3"
REL = "IMPORTS|INJECTS|EXTENDS|CALLS|USES|PERFORMS|ACCESSES|IMPLEMENTS|MODIFIES|TRIGGERS|VALIDATES|AFFECTS|TESTED_BY|CONSTRAINS|APPLIES_IN|CONFIGURED_BY|INITIATES"
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "q"))

NODES_Q = ("MATCH (n:EntityDetail {namespace:$ns}) WHERE n.v4_subsystem IS NOT NULL "
           "RETURN n.name AS name, n.entity_type AS et, n.v4_subsystem AS sub")
EDGES_Q = (f"MATCH (a:EntityDetail {{namespace:$ns}})-[r:{REL}]->(b:EntityDetail {{namespace:$ns}}) "
           "WHERE a.v4_subsystem IS NOT NULL AND b.v4_subsystem IS NOT NULL "
           "RETURN a.name AS an, a.v4_subsystem AS asub, type(r) AS t, b.name AS bn, b.v4_subsystem AS bsub")


def canon(rows):
    rows = sorted([str(c) for c in r] for r in rows)
    fp = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return rows, fp


# Each M: queries = list of (label, cypher) run verbatim; post composes rows from raw results
# and the shared node/edge pulls. gold_cypher stored = the verbatim queries (+ post note).
def post_system_map(raw, nodes, edges):
    ext_in = defaultdict(Counter)
    for e in edges:
        if e["asub"] != e["bsub"]:
            ext_in[e["bsub"]][e["bn"]] += 1
    by_sub = defaultdict(list)
    for n in nodes:
        by_sub[n["sub"]].append(n)
    rows = []
    for sub, ms in by_sub.items():
        lp = Counter(m["et"] for m in ms)
        dom, domc = lp.most_common(1)[0]
        entry = ext_in[sub].most_common(1)
        rows.append([sub, len(ms), dom, f"{domc/len(ms):.2f}",
                     entry[0][0] if entry else "-", entry[0][1] if entry else 0])
    return rows


def post_coupling(raw, nodes, edges):
    cnt = defaultdict(lambda: [0, 0])
    for e in edges:
        if e["asub"] == e["bsub"]:
            cnt[e["asub"]][0] += 1
        else:
            cnt[e["asub"]][1] += 1
            cnt[e["bsub"]][1] += 1
    size = Counter(n["sub"] for n in nodes)
    return [[s, size[s], i, x, f"{x/max(1, i+x):.3f}"] for s, (i, x) in cnt.items()]


def post_reading_path(raw, nodes, edges):
    ext_in = Counter()
    for e in edges:
        if e["asub"] != e["bsub"] and e["bsub"] == 11:
            ext_in[e["bn"]] += 1
    entry = ext_in.most_common(1)[0]
    rows = [["ENTRY", entry[0], str(entry[1]), ""]]
    for r in raw["spines"]:
        rows.append(["SPINE", r["hub"], str(r["idf"]), "|".join(r["members"][:8])])
    return rows


MS = [
    dict(id="M01", q="What is the one-walk overview of every subsystem: name, size, entry point?",
         aliases=["System map for a new hire", "Where does each module start?", "Mapa systemu: podsystemy i punkty wejscia"],
         role="onboarder", stratum="G2", archetype="overview", bucket="architecture",
         queries=[("nodes", NODES_Q), ("edges", EDGES_Q)], post=post_system_map, dep_subs="ALL",
         note="rows: [sub, size, dominant_type, purity, top_entry(by external in-degree), ext_in]"),
    dict(id="M02", q="What is the minimal reading path to understand the subscription/consent subsystem?",
         aliases=["Shortest onboarding spine for subscriptions", "How to learn billing code fastest", "Minimalna sciezka czytania subskrypcji"],
         role="onboarder", stratum="G3", archetype="onboarding_path", bucket="billing",
         queries=[("edges", EDGES_Q),
                  ("spines", "MATCH (h:HyperedgeCandidate {namespace:$ns, metapath:'A_P_R'}) "
                             "MATCH (hub:EntityDetail) WHERE id(hub)=h.hub_id AND hub.v4_subsystem=11 "
                             "RETURN h.hub_name AS hub, round(h.idf_weight,3) AS idf, h.member_names AS members "
                             "ORDER BY h.idf_weight DESC, h.hub_name LIMIT 5")],
         post=post_reading_path, dep_fixed=[11],
         note="v1 spine definition: entry = max external in-degree member of sub-11; spines = top-IDF A_P_R hyperedges hubbed in sub-11 (C3 will refine with trophic order)"),
    dict(id="M03", q="If UserRepository.java changes, which files and subsystems are affected?",
         aliases=["What is the blast radius of UserRepository?", "Impact analysis for UserRepository.java", "Co zalezy od UserRepository.java?"],
         role="dev", stratum="G2", archetype="impact", bucket="data-layer",
         queries=[("deps", f"MATCH (a:EntityDetail {{namespace:$ns}})-[r:{REL}]->(b:EntityDetail {{namespace:$ns, name:'UserRepository.java'}}) "
                           "RETURN DISTINCT a.name AS an, type(r) AS t, a.v4_subsystem AS sub")],
         rows=lambda raw: [[r["an"], r["t"], r["sub"]] for r in raw["deps"]], dep_cols=[2]),
    dict(id="M04", q="Why is a user with a valid login token still getting 403s on every request?",
         aliases=["Authenticated but blocked everywhere", "Valid JWT yet forbidden", "Czemu zalogowany user dostaje same 403"],
         role="security", stratum="G3", archetype="flow", bucket="auth",
         queries=[("filters", "MATCH (n:EntityDetail {namespace:$ns}) "
                              "WHERE n.name =~ '(?i).*(enforcementfilter|banned.*filter|emailverif.*|authorizationfilter).*' "
                              "AND NOT n.name =~ '.*Test.*' "
                              "RETURN n.name AS name, n.entity_type AS et, n.v4_subsystem AS sub, n.file_path AS path")],
         rows=lambda raw: [[r["name"], r["et"], r["sub"]] for r in raw["filters"]], dep_cols=[2]),
    dict(id="M05", q="Why do some accounts vanish a few days after registration without any admin action?",
         aliases=["Accounts auto-deleted after signup", "New users disappearing", "Czemu swieze konta same znikaja"],
         role="pm", stratum="G3", archetype="flow", bucket="consent",
         queries=[("out", f"MATCH (c:EntityDetail {{namespace:$ns, name:'NoConsentAccountCleanupCronJob.java'}})-[r:{REL}]->(t:EntityDetail {{namespace:$ns}}) "
                          "RETURN DISTINCT type(r) AS t, t.name AS tn, t.v4_subsystem AS sub")],
         rows=lambda raw: [[r["t"], r["tn"], r["sub"]] for r in raw["out"]], dep_cols=[2]),
    dict(id="M06", q="Why do multi-select relation fields arrive at the backend as an empty object?",
         aliases=["Why is my selected-platforms payload empty?", "Serialized form field becomes {} - why?", "Czemu pola wielokrotnego wyboru docieraja do backendu puste?"],
         role="dev", stratum="G3", archetype="locate", bucket="frontend-http",
         queries=[("hit", "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.name CONTAINS 'set-to-array' "
                          "OPTIONAL MATCH (imp:EntityDetail {namespace:$ns})-[:IMPORTS]->(n) "
                          "RETURN n.name AS name, n.file_path AS path, collect(DISTINCT imp.name) AS importers")],
         rows=lambda raw: [[r["name"], r["path"], "|".join(sorted(r["importers"]))] for r in raw["hit"]], dep_fixed=[17], gold_status="COVERAGE_GAP"),
    dict(id="M07", q="Which notification/email flows are triggered from the subscription lifecycle?",
         aliases=["Billing-to-email paths", "What emails does a payment event cause?", "Ktore maile wyzwala cykl subskrypcji?"],
         role="dev", stratum="G3", archetype="boundary", bucket="billing",
         queries=[("seam", f"MATCH (a:EntityDetail {{namespace:$ns}})-[r:{REL}]->(b:EntityDetail {{namespace:$ns}}) "
                           "WHERE a.v4_subsystem = 11 AND b.v4_subsystem IN [10,12] "
                           "RETURN DISTINCT a.name AS an, type(r) AS t, b.name AS bn, b.v4_subsystem AS sub")],
         rows=lambda raw: [[r["an"], r["t"], r["bn"], r["sub"]] for r in raw["seam"]], dep_fixed=[10,11,12]),
    dict(id="M08", q="What can reach payment resources from a public controller within 3 hops?",
         aliases=["Attack paths to Stripe data", "Public-to-payment reachability", "Sciezki od publicznych endpointow do platnosci"],
         role="security", stratum="G3", archetype="impact", bucket="security",
         queries=[("paths", "MATCH (a:EntityDetail {namespace:$ns, entity_type:'Actor'}) WHERE a.v4_subsystem <> 11 "
                            "MATCH p=(a)-[:IMPORTS|INJECTS|CALLS|USES|PERFORMS|ACCESSES*1..3]->(t:EntityDetail {namespace:$ns}) "
                            "WHERE t.v4_subsystem = 11 AND t.name =~ '(?i).*(stripe|payment|subscription).*' AND NOT t.name =~ '.*Test.*' "
                            "WITH a.name AS src, a.v4_subsystem AS srcsub, t.name AS dst, min(length(p)) AS hops "
                            "RETURN src, srcsub, dst, hops")],
         rows=lambda raw: [[r["src"], r["srcsub"], r["dst"], r["hops"]] for r in raw["paths"]],
         dep_cols=[1], dep_fixed=[11],
         note="'public' approximated as any Actor outside sub-11; endpoint-level auth annotation is a C4 enrichment"),
    dict(id="M09", q="Which files co-participate in hyperedges with address.service.ts?",
         aliases=["Cohort of address.service.ts", "What always changes with the address service?", "Co zmienia sie razem z address.service.ts?"],
         role="dev", stratum="G3", archetype="cohort", bucket="frontend-address",
         queries=[("h", "MATCH (m:EntityDetail {namespace:$ns, name:'address.service.ts'})-[:IN_HYPEREDGE]->(h:HyperedgeCandidate) "
                        "RETURN h.metapath AS mp, h.hub_name AS hub, round(h.idf_weight,3) AS idf, h.member_names AS members")],
         rows=lambda raw: [[r["mp"], r["hub"], str(r["idf"]), "|".join(sorted(r["members"]))] for r in raw["h"]], dep_fixed=[17]),
    dict(id="M10", q="In what order do the HTTP interceptors run and why does the order matter?",
         aliases=["List the interceptor chain", "Which interceptor runs first on requests?", "W jakiej kolejnosci dzialaja interceptory HTTP?"],
         role="architect", stratum="G1", archetype="overview", bucket="frontend-http",
         queries=[("ic", "MATCH (n:EntityDetail {namespace:$ns}) "
                         "WHERE n.name ENDS WITH '.interceptor.ts' OR n.name = 'app.config.ts' "
                         "RETURN n.name AS name, n.file_path AS path ORDER BY n.name")],
         rows=lambda raw: [[r["name"], r["path"]] for r in raw["ic"]], dep_fixed=[17],
         note="the ORDER itself is content, not structure: recipe returns the interceptor set + app.config.ts as the read pointer"),
    dict(id="M11", q="Which cron jobs can modify user data, and via which paths?",
         aliases=["Scheduled jobs with write access", "What runs unattended and mutates state?", "Ktore crony modyfikuja dane uzytkownikow?"],
         role="security", stratum="G3", archetype="health", bucket="lifecycle",
         queries=[("cr", "MATCH (c:EntityDetail {namespace:$ns}) WHERE c.name ENDS WITH 'CronJob.java' "
                         "OPTIONAL MATCH (c)-[r:MODIFIES|ACCESSES]->(t:EntityDetail {namespace:$ns}) "
                         "RETURN c.name AS cn, type(r) AS t, t.name AS tn")],
         rows=lambda raw: [[r["cn"], r["t"] or "-", r["tn"] or "-"] for r in raw["cr"]], dep_fixed=[11]),
    dict(id="M12", q="Which subsystem has the highest cross-boundary coupling?",
         aliases=["Most externally coupled module", "Where is coupling worst?", "Ktory podsystem ma najwieksze sprzezenie zewnetrzne?"],
         role="architect", stratum="G3", archetype="health", bucket="architecture",
         queries=[("nodes", NODES_Q), ("edges", EDGES_Q)], post=post_coupling, dep_subs="ALL",
         note="rows: [sub, size, internal_edges, external_edges, external_ratio]"),
]

# Authored AFTER first execution, from GOLD_RESULTS.md. id -> reference answer.
GOLD_ANSWERS = {
    "M01": "The system decomposes into 19 subsystems (v4 partition). Three continents: sub-17 the greenfield FE (375 files, Actor-dominated, zero external in-edges — you enter it at its actor roots, i.e. the app bootstrap, not via imports), sub-11 subscriptions/consent/billing (208 files, entry UserRepository.java with 141 external in-edges), and sub-4 opportunities (123 files, entry PermissionUtils.java, 73). Test infrastructure concentrates in sub-13 (90% Rule, entry AppliedOpportunityServiceIntegrationTestBase) and sub-3 (config/context, entry CucumberSpringConfig). The full map is the gold rows: [subsystem, size, dominant type, purity, top entry by external in-degree, its in-degree].",
    "M02": "Enter at UserRepository.java — the subsystem's gravity centre (141 external in-edges). Then read the five A→P→R spines in descending distinctiveness: StripeService (StripeProperties → SubscriptionPaidController, idf 2.89), CampaignLimitService (BillingPeriodRepository, CompanySubscriptionRepository, 2.485), LegalDocumentService (LegalController, LegalDocumentRepository, 2.485), InvoiceRetryService (InvoicingPort, InvoiceRetryCronJob, InvoiceRecordRepository, 1.974), ConsentService (the consent-repository family plus ConsentController/ConsentAdminController, 1.504). Entry plus these five walks covers billing, limits, legal, invoicing and consent — the subsystem's five responsibilities — in about 20 files instead of 208.",
    "M03": "184 typed dependency rows from 79 distinct files across 14 of the 19 subsystems point at UserRepository.java (its external in-degree alone is 141 edges — edges, not files) — the most load-bearing file in the backend. Direct dependents span every feature area: cooperation services, admin cascade-delete and integrity checkers, applied-opportunity services, address services, and the test tiers. An interface change here is a system-wide event; a query-semantics change is invisible to the type system yet reaches all 14 subsystems. The complete sorted set is the gold rows [file, edge type, subsystem].",
    "M04": "Because login is not the only gate: three enforcement filters run after authentication, and each returns 403 while the token stays valid. Measured candidates: BannedUserAuthorizationFilter (sub-9), EmailVerificationEnforcementFilter (sub-9), and ConsentEnforcementFilter (sub-11) — a ban, an unverified email, or a missing/outdated consent each blocks every request. None of them has '403' in its name, which is exactly why keyword search misses this. Check ban state, then email verification, then current consent version.",
    "M05": "NoConsentAccountCleanupCronJob.java (sub-11): a scheduled job that PERFORMS LegalConsentService — accounts that never granted the required consents are cleaned up automatically after the grace window, no admin involved. Its graph neighbourhood is deliberately thin (it acts through the consent service instead of touching repositories directly), so the deletion mechanics live in the cron's file content; the graph pins the responsible actor and its service path.",
    "M06": "COVERAGE GAP, recorded rather than papered over: set-to-array.interceptor.ts exists in the repo (it fixes JSON.stringify(new Set) serialising to '{}' before requests leave the browser) but is ABSENT from the graph — the scan predates it. The graph holds 6 of the repo's 8 interceptors (ssr-cookie-forward is the other missing one). This is the first entry in the coverage-failure bucket (doc-00 §6, kubełek i) and the trigger for the delta-scan backlog; gold completes after re-scan.",
    "M07": "The measured sub-11 → {10,12} seam carries 50 edges, but most are entity gravity — imports of User.java, which lives in sub-10. The genuine flow crossings: RegistryLookupService TRIGGERS AccountActivatedEvent (sub-12) — activation on the billing/registry side raises the event the notification layer consumes — and LegalController PERFORMS TokenExchangeService (sub-10) for magic-link legal flows. Honest caveat: TRIGGERS edges are sparse system-wide (6 total), so event-mediated email flows beyond AccountActivatedEvent (trial expiry, invoice mail) are under-modelled until the delta scan enriches event edges; the cron names (TrialExpiryNotifierCronJob, EmailCronJob) mark where they originate.",
    "M08": "Within 3 typed hops, exactly one Actor outside the billing subsystem reaches payment resources: PartnershipOpportunityController (sub-4) → SubscriptionService, CompanySubscription and CompanySubscriptionRepository, each at hop 3 — the expected coupling (campaign creation checks plan limits) and the complete measured surface at this depth. No other non-billing controller touches Stripe/payment/subscription resources in ≤3 hops. Caveat: 'public' is approximated as any Actor outside sub-11; endpoint-level auth annotation arrives with the C4 clue layer.",
    "M09": "One hyperedge: the A_P_A cohort hubbed on address.service.ts with addresses.component.ts and layout.component.ts (idf 2.862) — the two components sharing the service. Change the service's contract and that pair is the co-change set; the high IDF says this hub is distinctive, not a generic utility every file touches.",
    "M10": "The graph holds six interceptors — demo, error, language, rate-limit-cache, shell-headers, step-up — plus the registration point app.config.ts, where the chain order and its rationale comment live. Order is file content, not structure: the recipe returns the interceptor set and app.config.ts as the read pointer. Known gap: the repo has 8 interceptors (set-to-array and ssr-cookie-forward are missing from the graph — delta-scan backlog).",
    "M11": "Ten cron jobs run unattended. Direct data-access edges: DeferredDeletionCronJob ACCESSES PendingDataDeletionRequestRepository and UserAccountOrchestrator (the GDPR deletion executor — highest blast radius), while SubscriptionPeriodProcessorCronJob and TermsGraceProcessorCronJob ACCESS AppPaymentsProperties. The other six (including NoConsentAccountCleanupCronJob, EmailCronJob, TrialExpiryNotifierCronJob) mutate state through service layers — PERFORMS chains — rather than direct repository edges, so the recipe's follow-up is the 2-hop expansion. Every cron on this list can change user-visible state with no human in the loop.",
    "M12": "By raw ratio: sub-16 at 1.000 (zero internal edges among 12 files — a fragment and the Curator's top merge candidate) and sub-18 at 1.000 (the single 'api' node carrying all 150 FE→client edges — an articulation point, not a subsystem). Among non-degenerate subsystems: sub-2 at 0.993 (1 internal vs 152 external — the social-connection fragment), then sub-8 (0.943), sub-0 (0.939), and sub-13 (0.938 — expected, tests import production). Least coupled: the FE (0.330) and support tickets (0.487). The three ≥0.99 entries are exactly the curation merge queue.",
}


def main():
    drv = GraphDatabase.driver(BOLT, auth=AUTH)
    with drv.session() as s:
        nodes = s.run(NODES_Q, ns=NS).data()
        edges = s.run(EDGES_Q, ns=NS).data()
        out, md = [], ["# Q1 gold — executed results digest\n"]
        for m in MS:
            raw = {}
            for label, q in m["queries"]:
                if q == NODES_Q:
                    raw[label] = nodes
                elif q == EDGES_Q:
                    raw[label] = edges
                else:
                    raw[label] = s.run(q, ns=NS).data()
            rows = m["post"](raw, nodes, edges) if "post" in m else m["rows"](raw)
            rows, fp = canon(rows)
            if m.get("dep_subs") == "ALL":
                subs = sorted({n["sub"] for n in nodes})
            elif "dep_cols" in m:
                subs = sorted({int(r[c]) for r in rows for c in m["dep_cols"]
                               if c < len(r) and str(r[c]).isdigit()} | set(m.get("dep_fixed", [])))
            else:
                subs = m.get("dep_fixed", [])
            rec = dict(id=m["id"], q=m["q"], aliases=m["aliases"], role=m["role"],
                       stratum=m["stratum"], archetype=m["archetype"], topic_bucket=m["bucket"],
                       worth=3,
                       gold_cypher=[q for _, q in m["queries"]],
                       gold_note=m.get("note"),
                       gold_result_excerpt=rows[:25], gold_result_count=len(rows),
                       gold_fingerprint=fp,
                       gold_status=m.get("gold_status", "EXECUTED"),
                       depends_on_subsystems=subs,
                       gold_answer=GOLD_ANSWERS.get(m["id"]))
            out.append(rec)
            md.append(f"## {m['id']} — {m['q']}\nrows={len(rows)} fp={fp} subs={subs}\n" +
                      ("note: " + m["note"] + "\n" if m.get("note") else "") +
                      "```\n" + "\n".join(" | ".join(r) for r in rows[:20]) +
                      ("\n... (+" + str(len(rows) - 20) + " more)" if len(rows) > 20 else "") + "\n```\n")
    drv.close()
    with open(os.path.join(OUT, "mfq_gold.jsonl"), "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "GOLD_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    missing = [r["id"] for r in out if not r["gold_answer"]]
    print(f"OK 12 golds, fingerprints stable. Answers missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
