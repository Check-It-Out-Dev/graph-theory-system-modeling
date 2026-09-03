# CodeMap Q1-gold FULL: all 104 bank questions -> executed gold via the 9 recipe archetypes.
# The recipe functions here are the REFERENCE IMPLEMENTATION of the future MCP recipe layer
# (D5: recipe selection over query generation) — schema/SCHEMA.md section 5.
#
# Flow: manifest maps each bank question (raw/*.jsonl, ids BE01.., FE01.., GR01..) to a
# recipe + params. Execute -> canonical rows -> fingerprint -> render answer (auto template
# per recipe, optionally prefixed by a HAND insight). Questions the graph cannot answer are
# classified honestly: CONTENT_POINTER (graph pins WHERE, content holds the answer) or
# COVERAGE_GAP (graph misses the artifact; delta-scan backlog). 12 bank questions are
# dup_of the M01-M12 exemplars (mfq_gold.jsonl) and inherit their gold.
#
# Usage: PYTHONUTF8=1 python q_gold_all.py   (idempotent; writes ../q/mfq_all.jsonl + digest)

import argparse, hashlib, json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = os.path.abspath(os.path.join(HERE, "..", "q"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "graph", "authoring")))

# Ladybug (MIT) is the DEFAULT authoring backend since 2026-09-02; --backend neo4j is
# kept only for the migration differential (both backends must yield identical gold
# fingerprints — the 104-question battery is the acceptance gate).
BOLT, AUTH, NS = "bolt://127.0.0.1:7611", ("neo4j", "password"), "CheckItOutV3"
REL = "IMPORTS|INJECTS|EXTENDS|CALLS|USES|PERFORMS|ACCESSES|IMPLEMENTS|MODIFIES|TRIGGERS|VALIDATES|AFFECTS|TESTED_BY|CONSTRAINS|APPLIES_IN|CONFIGURED_BY|INITIATES"


def canon(rows):
    rows = sorted([str(c) for c in r] for r in rows)
    fp = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return rows, fp


def topn(counter, n):
    """DETERMINISM LAW (2026-09-02, Ladybug migration): every top-N in a recipe breaks
    ties by name, never by backend return order. The migration differential exposed the
    class — identical result SETS, different top-N cutoffs and enumeration order between
    Neo4j and Ladybug. A recipe that depends on return order was never deterministic;
    it was per-backend lucky."""
    return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:n]


class Graph:
    """One pull of nodes+edges; recipes compute locally (1374 nodes — RAM beats round-trips)."""

    def __init__(self, session):
        # nid/aid/bid (2026-09-02, task 63 follow-up): file NAME is not a vertex identity —
        # the graph holds two distinct files per name (UnifiedStorageConfiguration.java,
        # HashingUtilUnitTest.java, both in sub-3). Keying an adjacency by name silently
        # merges them into one vertex, which lost 2 vertices and distorted 66 of sub-3's 129
        # heights. r_trophic_inversion keys by nid; other recipes are unchanged.
        self.nodes = session.run(
            "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.v4_subsystem IS NOT NULL "
            "RETURN id(n) AS nid, n.name AS name, n.entity_type AS et, "
            "n.v4_subsystem AS sub, n.file_path AS path",
            ns=NS).data()
        self.edges = session.run(
            f"MATCH (a:EntityDetail {{namespace:$ns}})-[r:{REL}]->(b:EntityDetail {{namespace:$ns}}) "
            "WHERE a.v4_subsystem IS NOT NULL AND b.v4_subsystem IS NOT NULL "
            "RETURN id(a) AS aid, a.name AS an, a.v4_subsystem AS asub, type(r) AS t, "
            "id(b) AS bid, b.name AS bn, b.v4_subsystem AS bsub",
            ns=NS).data()
        self.hyper = session.run(
            "MATCH (h:HyperedgeCandidate {namespace:$ns}) MATCH (hub:EntityDetail) "
            "WHERE id(hub) = h.hub_id "
            "RETURN h.metapath AS mp, h.hub_name AS hub, hub.v4_subsystem AS hubsub, "
            "round(h.idf_weight,3) AS idf, h.member_names AS members", ns=NS).data()
        self.viol = session.run(
            "MATCH (a:EntityDetail {namespace:$ns})-[r:ALGEBRA_VIOLATION]->(b:EntityDetail {namespace:$ns}) "
            "RETURN a.name AS an, a.v4_subsystem AS asub, b.name AS bn, b.v4_subsystem AS bsub", ns=NS).data()
        self._index()

    def _index(self):
        # DETERMINISM AT THE BOUNDARY: backend scan order is not a contract (Ladybug's
        # parallel scans vary run-to-run; Neo4j's order is merely habitual). Sort ONCE
        # here so every recipe iteration downstream is reproducible by construction.
        self.nodes.sort(key=lambda n: n["nid"])
        self.edges.sort(key=lambda e: (e["aid"], e["bid"], e["t"]))
        self.hyper.sort(key=lambda h: (str(h["mp"]), str(h["hub"])))
        self.viol.sort(key=lambda v: (str(v["an"]), str(v["bn"])))
        self.by_name = {n["name"]: n for n in self.nodes}
        self.in_e, self.out_e = defaultdict(list), defaultdict(list)
        for e in self.edges:
            self.out_e[e["an"]].append(e)
            self.in_e[e["bn"]].append(e)

    def find(self, rx):
        import re
        p = re.compile(rx, re.I)
        return sorted((n for n in self.nodes if p.search(n["name"])),
                      key=lambda n: (n["name"], n["nid"]))  # determinism law


class LadybugGraph(Graph):
    """Same in-RAM shape, pulled from the Ladybug authoring store (the default)."""

    def __init__(self, store):
        self.nodes = store.q(
            "MATCH (n:Entity) WHERE n.sub IS NOT NULL RETURN n.nid AS nid, "
            "n.name AS name, n.entity_type AS et, n.sub AS sub, n.file_path AS path")
        self.edges = store.q(
            "MATCH (a:Entity)-[r:Dep]->(b:Entity) "
            "WHERE a.sub IS NOT NULL AND b.sub IS NOT NULL "
            "RETURN a.nid AS aid, a.name AS an, a.sub AS asub, r.rel AS t, "
            "b.nid AS bid, b.name AS bn, b.sub AS bsub")
        hyper = store.q(
            "MATCH (h:Hyperedge) RETURN h.metapath AS mp, h.hub_name AS hub, "
            "h.hubsub AS hubsub, h.idf AS idf, h.members AS members")
        for h in hyper:  # members travel as a JSON string in the store
            h["members"] = json.loads(h["members"] or "null")
            h["idf"] = round(h["idf"], 3) if h["idf"] is not None else None
        self.hyper = hyper
        self.viol = store.q(
            "MATCH (a:Entity)-[:Violation]->(b:Entity) RETURN a.name AS an, "
            "a.sub AS asub, b.name AS bn, b.sub AS bsub")
        self._index()


# ---------------- recipes (reference implementation of SCHEMA.md section 5) --------------

def r_locate(g, rx, note=""):
    hits = g.find(rx)
    rows = [[h["name"], h["et"], h["sub"], h["path"] or ""] for h in hits]
    if not hits:
        return rows, "NOT IN GRAPH — no node matches; see gold_status.", True
    top = hits[: min(6, len(hits))]
    ans = "; ".join(f"{h['name']} ({h['et']}, sub-{h['sub']})" for h in top)
    more = f" (+{len(hits)-6} more)" if len(hits) > 6 else ""
    return rows, f"Located: {ans}{more}. Paths in gold rows.{(' ' + note) if note else ''}", False


def r_impact(g, name):
    deps = g.in_e.get(name, [])
    rows = [[e["an"], e["t"], e["asub"]] for e in deps]
    files = {e["an"] for e in deps}
    subs = sorted({e["asub"] for e in deps})
    top = topn(Counter(e["an"] for e in deps), 5)
    ans = (f"{len(deps)} dependency edges from {len(files)} files across {len(subs)} subsystems point at {name}. "
           f"Heaviest dependents: {', '.join(n for n, _ in top)}. Full sorted set in gold rows.")
    return rows, ans, len(deps) == 0


def r_flow(g, rx, hops=2):
    starts = [n["name"] for n in g.find(rx)]
    seen, frontier, rows = set(starts), list(starts), []
    for h in range(hops):
        nxt = []
        for s in frontier:
            for e in g.out_e.get(s, []):
                rows.append([e["an"], e["t"], e["bn"], e["bsub"], h + 1])
                if e["bn"] not in seen:
                    seen.add(e["bn"])
                    nxt.append(e["bn"])
        frontier = nxt
    kinds = Counter(r[1] for r in rows)
    ans = (f"Flow from {', '.join(starts[:3])}{'...' if len(starts) > 3 else ''}: "
           f"{len(rows)} typed steps within {hops} hops touching {len(seen)-len(starts)} downstream files "
           f"({', '.join(f'{k} {v}' for k, v in kinds.most_common(4))}). Walk the gold rows in hop order.")
    return rows, ans, not starts


def r_boundary(g, sub_a, sub_b):
    rows = [[e["an"], e["t"], e["bn"], f"{e['asub']}->{e['bsub']}"] for e in g.edges
            if (e["asub"], e["bsub"]) in ((sub_a, sub_b), (sub_b, sub_a))]
    kinds = Counter(r[1] for r in rows)
    notable = [r for r in rows if r[1] not in ("IMPORTS",)]
    ans = (f"Seam sub-{sub_a} <-> sub-{sub_b}: {len(rows)} edges ({', '.join(f'{k} {v}' for k, v in kinds.most_common())}). "
           f"Beyond imports: {'; '.join(f'{r[0]} {r[1]} {r[2]}' for r in notable[:5]) or 'none — the seam is pure imports'}.")
    return rows, ans, not rows


def r_cohort(g, name=None, metapath=None):
    hs = sorted((h for h in g.hyper
                 if (name is None or name == h["hub"] or name in (h["members"] or []))
                 and (metapath is None or h["mp"] == metapath)),
                key=lambda h: (str(h["mp"]), str(h["hub"])))  # determinism law
    rows = [[h["mp"], h["hub"], f"sub-{h['hubsub']}", str(h["idf"]), "|".join(sorted(h["members"] or []))] for h in hs]
    ans = (f"{len(hs)} hyperedge cohort(s): "
           + "; ".join(f"{h['mp']} hub {h['hub']} (idf {h['idf']}) with {len(h['members'] or [])} members" for h in hs[:5])
           + ". Members co-change with the hub; IDF weights distinctiveness.")
    return rows, ans, not hs


def r_health(g, kind):
    if kind == "violations":
        pairs = Counter((e["asub"], e["bsub"]) for e in g.viol)
        rows = [[a, b, c] for (a, b), c in pairs.items()]
        top = topn(pairs, 5)
        ans = (f"{len(g.viol)} ALGEBRA_VIOLATION edges across {len(pairs)} subsystem pairs. "
               f"Hotspots: {', '.join(f'sub-{a}->sub-{b} ({c})' for (a, b), c in top)}. "
               "These are typed-layer rule breaks — curation reviews the top pairs first.")
    elif kind == "injects_cross":
        cross = [e for e in g.edges if e["t"] == "INJECTS" and e["asub"] != e["bsub"]]
        pairs = Counter((e["asub"], e["bsub"]) for e in cross)
        rows = [[a, b, c] for (a, b), c in pairs.items()]
        ans = (f"{len(cross)} cross-subsystem INJECTS edges over {len(pairs)} pairs — DI reaching across boundaries. "
               f"Top: {', '.join(f'sub-{a}->sub-{b} ({c})' for (a, b), c in topn(pairs, 5))}.")
    elif kind == "untested":
        tested = {e["bsub"] for e in g.edges if e["t"] == "TESTED_BY"} | {e["asub"] for e in g.edges if e["t"] == "TESTED_BY"}
        sizes = Counter(n["sub"] for n in g.nodes)
        rows = [[s, sizes[s], "tested" if s in tested else "NO_TEST_EDGES"]
                for s in sorted(sizes, key=str)]
        untested = sorted(s for s in sizes if s not in tested)
        ans = (f"Subsystems with zero TESTED_BY coupling: {untested or 'none'}. "
               "Caveat: TESTED_BY edges (104 total) undercount real coverage — most test linkage rides IMPORTS from sub-13/sub-3; "
               "treat this as a shape signal, not a coverage report.")
    elif kind == "top_indegree":
        cnt = Counter(e["bn"] for e in g.edges)
        rows = [[n, c, g.by_name[n]["sub"]] for n, c in topn(cnt, 15)]
        ans = ("Most load-bearing files by dependency in-degree: "
               + ", ".join(f"{n} ({c}, sub-{s})" for n, c, s in rows[:8]) + ".")
    elif kind == "bridges":
        seam_carriers = defaultdict(Counter)
        for e in g.edges:
            if e["asub"] != e["bsub"]:
                seam_carriers[(e["asub"], e["bsub"])][e["bn"]] += 1
        rows = []
        for (a, b), c in sorted(seam_carriers.items(), key=str):
            total = sum(c.values())
            top_n, top_c = topn(c, 1)[0]
            if total >= 10 and top_c / total >= 0.9:
                rows.append([f"{a}->{b}", top_n, top_c, total])
        ans = ("Single-file seam carriers (>=90% of a >=10-edge seam through one node): "
               + "; ".join(f"{r[1]} carries {r[2]}/{r[3]} of {r[0]}" for r in rows[:6])
               + ". Removing such a file disconnects the subsystems — structural single points of failure.")
    elif kind == "hubs":
        multi = []
        for h in sorted(g.hyper, key=lambda x: (str(x["mp"]), str(x["hub"]))):
            subs = {g.by_name[m]["sub"] for m in (h["members"] or []) if m in g.by_name} | {h["hubsub"]}
            if len(subs) > 1:
                multi.append((h, subs))
        rows = [[h["mp"], h["hub"], str(h["idf"]), ",".join(map(str, sorted(s)))] for h, s in multi]
        ans = (f"{len(multi)} hyperedge hubs bridge multiple subsystems: "
               + "; ".join(f"{h['hub']} ({h['mp']}, subs {sorted(s)})" for h, s in multi[:6]) + ".")
    elif kind == "config_reach":
        cfg = [n for n in g.nodes if n["name"].endswith("Properties.java") or n["et"] == "Context"]
        cnt = [(n["name"], len(g.in_e.get(n["name"], [])), n["sub"]) for n in cfg]
        cnt.sort(key=lambda x: (-x[1], x[0]))  # determinism law: name breaks ties
        rows = [[a, b, c] for a, b, c in cnt[:15]]
        ans = ("Highest-reach configuration by in-degree: "
               + ", ".join(f"{n} ({d} deps, sub-{s})" for n, d, s in cnt[:6]) + ".")
    elif kind == "events":
        evs = sorted((n for n in g.nodes if n["et"] == "Event"), key=lambda n: n["name"])
        rows = []
        for ev in evs:
            producers = sorted({e["an"] for e in g.in_e.get(ev["name"], []) if e["t"] == "TRIGGERS"})
            touchers = sorted({e["an"] for e in g.in_e.get(ev["name"], []) if e["t"] != "TRIGGERS"})
            rows.append([ev["name"], ev["sub"], "|".join(producers) or "-", "|".join(touchers[:6]) or "-"])
        ans = (f"{len(evs)} Event nodes exist. "
               + "; ".join(f"{r[0]} (sub-{r[1]}; triggered by {r[2]})" for r in rows)
               + ". Caveat: TRIGGERS edges are sparse (6 system-wide) — consumers are under-modelled until delta enrichment.")
    return rows, ans, not rows


def r_onboarding(g, sub):
    ext_in, int_in = Counter(), Counter()
    for e in g.edges:
        if e["bsub"] == sub:
            (ext_in if e["asub"] != sub else int_in)[e["bn"]] += 1
    rows = [["ENTRY", n, str(c), ""] for n, c in topn(ext_in, 5)]
    roots = sorted(n["name"] for n in g.nodes if n["sub"] == sub and n["et"] == "Actor"
                   and int_in[n["name"]] == 0 and not ext_in[n["name"]])[:5]
    rows += [["ACTOR_ROOT", r, "", ""] for r in roots]
    for h in sorted((h for h in g.hyper if h["mp"] == "A_P_R" and h["hubsub"] == sub),
                    key=lambda x: (-(x["idf"] or 0), str(x["hub"])))[:5]:
        rows.append(["SPINE", h["hub"], str(h["idf"]), "|".join((h["members"] or [])[:8])])
    ans = (f"Sub-{sub} entry: external in-edges point at {', '.join(n for n, _ in topn(ext_in, 3)) or 'nothing (a sink — use actor roots)'}; "
           f"actor roots: {', '.join(roots) or 'none'}; spines (A_P_R, IDF-ordered) in gold rows.")
    return rows, ans, not rows


# ---------------- manifest -----------------------------------------------------------

# (bank_file, line_idx(0-based)) resolved at load; entries keyed BEnn/FEnn/GRnn in file order.
R = dict  # alias for brevity
MANIFEST = {
    # --- BE (qminer_be.jsonl, 37) ---
    "BE01": R(recipe="locate", rx="PaymentsDisabledBootGuard", hand=True),
    "BE02": R(recipe="locate", rx="CronJob\\.java$", note="Schedules live in each class's @Scheduled/@SchedulerLock annotations — content, not structure."),
    "BE03": R(recipe="locate", rx="application(-[a-z0-9-]+)?\\.ya?ml"),
    "BE04": R(recipe="locate", rx="SubscriptionPlan|SubscriptionStatus"),
    "BE05": R(recipe="locate", rx="Fakturownia|InvoicingPort"),
    "BE06": R(recipe="locate", rx="ConsentEnforcementFilter", hand=True),
    "BE07": R(recipe="locate", rx="RateLimit"),
    "BE08": R(recipe="locate", rx="StepUp"),
    "BE09": R(recipe="flow", rx="AdminCascadeDelete", hops=1),
    "BE10": R(recipe="locate", rx="Instagram"),
    "BE11": R(recipe="locate", rx="TermsVersion|LegalController|LegalAdminController|LegalDocument"),
    "BE12": R(recipe="locate", rx="AppLanguage|LocaleResolver", hand=True),
    "BE13": R(recipe="content", pointer="checkItOut-be2/LOCAL_DATABASE_SETUP.md", hand=True),
    "BE14": R(recipe="content", pointer="checkItOut-be2/docs/security/pentest-remediation-2026-09.md", hand=True),
    "BE15": R(recipe="content", pointer="checkItOut-be2/docs/STORAGE_CLEANUP_STRATEGY.md", hand=True),
    "BE16": R(recipe="flow", rx="^SubscriptionService\\.java$", hops=1, hand=True),
    "BE17": R(recipe="flow", rx="InvoiceRetry", hops=1, hand=True),
    "BE18": R(recipe="locate", rx="FilterRegistrationConfiguration|Filter\\.java$", note="The ORDER is declared in FilterRegistrationConfiguration — content; the graph pins the set."),
    "BE19": R(recipe="flow", rx="TrialExpiryNotifier", hops=2),
    "BE20": R(recipe="locate", rx="CronJob\\.java$", hand=True),
    "BE21": R(recipe="flow", rx="TermsGraceProcessor", hops=2, hand=True),
    "BE22": R(recipe="locate", rx="Redis|UserCacheService", hand=True),
    "BE23": R(recipe="flow", rx="BannedUser", hops=1, hand=True),
    "BE24": R(recipe="flow", rx="CampaignLimitService\\.java$", hops=1, hand=True),
    "BE25": R(recipe="flow", rx="DeferredDeletionCronJob", hops=2, hand=True),
    "BE26": R(recipe="locate", rx="GeoIp|GeoLocation"),
    "BE27": R(recipe="locate", rx="NotificationEventListener|AppliedOpportunity.*Service", hand=True),
    "BE28": R(recipe="locate", rx="EmailCronJob|AppLanguage", hand=True),
    "BE29": R(recipe="locate", rx="service-account|keystore", hand=True),
    "BE30": R(dup_of="M-style", recipe="locate", rx="PaymentsDisabledBootGuard", hand=True),
    "BE31": R(recipe="flow", rx="CampaignLimit", hops=1, hand=True),
    "BE32": R(dup_of="M04"),
    "BE33": R(dup_of="M05"),
    "BE34": R(recipe="flow", rx="InvoiceRetryCronJob", hops=1, hand=True),
    "BE35": R(recipe="content", pointer="entity classes (all @Version)", hand=True),
    "BE36": R(recipe="locate", rx="RequestCorrelation", hand=True),
    "BE37": R(recipe="locate", rx="StorageUrlValidator", hand=True),
    # --- FE (qminer_fe.jsonl, 34) ---
    "FE01": R(recipe="locate", rx="^app\\.routes\\.ts$"),
    "FE02": R(recipe="locate", rx="sign-in\\.component"),
    "FE03": R(recipe="content", pointer="src/assets/i18n/{en,pl}.json + check:i18n-parity gate", hand=True),
    "FE04": R(recipe="content", pointer="app.config.ts (Transloco: en/pl, defaultLang pl)", hand=True),
    "FE05": R(recipe="content", pointer="app.config.ts iter-107 comment", hand=True),
    "FE06": R(recipe="flow", rx="error\\.interceptor", hops=1, hand=True),
    "FE07": R(recipe="locate", rx="step-up"),
    "FE08": R(dup_of="M10"),
    "FE09": R(dup_of="M06"),
    "FE10": R(recipe="locate", rx="ApiConfiguration|api-configuration", hand=True),
    "FE11": R(recipe="locate", rx="proxy\\.conf|api-configuration|ApiConfiguration", hand=True),
    "FE12": R(recipe="content", pointer="package.json scripts openapi:gen / openapi:cycle", hand=True),
    "FE13": R(recipe="flow", rx="company-setup|registry\\.service", hops=1),
    "FE14": R(recipe="content", pointer="app.routes.ts public routes", hand=True),
    "FE15": R(recipe="locate", rx="technical-survey|survey"),
    "FE16": R(recipe="content", pointer="app.routes.ts redirect entries", hand=True),
    "FE17": R(recipe="locate", rx="settings"),
    "FE18": R(recipe="locate", rx="(?i)seo.*title|title.*strategy"),
    "FE19": R(recipe="locate", rx="visual-parity|component-pairs"),
    "FE20": R(recipe="content", pointer="playwright.config.ts device projects", hand=True),
    "FE21": R(recipe="locate", rx="(?i)bdd|\\.feature"),
    "FE22": R(recipe="locate", rx="(?i)trace|canonicalize", hand=True),
    "FE23": R(recipe="locate", rx="real-login"),
    "FE24": R(recipe="locate", rx="(?i)sandbox"),
    "FE25": R(recipe="locate", rx="demo\\.interceptor|demo-fixtures|scenario-registry"),
    "FE26": R(recipe="locate", rx="ssr-cookie-forward", hand=True),  # expected COVERAGE_GAP
    "FE27": R(recipe="content", pointer="app.config.ts provideClientHydration(withEventReplay())", hand=True),
    "FE28": R(recipe="flow", rx="shell-headers\\.interceptor", hops=1, hand=True),
    "FE29": R(recipe="locate", rx="rate-limit"),
    "FE30": R(recipe="content", pointer="package.json check:* scripts", hand=True),
    "FE31": R(recipe="locate", rx="(?i)error.*component|not-found|404"),
    "FE32": R(recipe="locate", rx="layout"),
    "FE33": R(recipe="locate", rx="api-frozen|subscription\\.client|hidden-models", hand=True),
    "FE34": R(recipe="locate", rx="(?i)collaboration|campaign"),
    # --- GR (qminer_graph.jsonl, 33) ---
    "GR01": R(dup_of="M03"),
    "GR02": R(recipe="impact", name="AccountStatus.java"),
    "GR03": R(recipe="impact", name="StripeService.java"),
    "GR04": R(dup_of="M12"),
    "GR05": R(recipe="boundary", a=17, b=18),
    "GR06": R(recipe="health", kind="violations"),
    "GR07": R(recipe="health", kind="injects_cross"),
    "GR08": R(dup_of="M02"),
    "GR09": R(recipe="onboarding", sub=17),
    "GR10": R(recipe="flow", rx="Consent.*Controller", hops=1),
    "GR11": R(dup_of="M11"),
    "GR12": R(recipe="boundary", a=13, b=4),
    "GR13": R(recipe="health", kind="untested"),
    "GR14": R(dup_of="M09"),
    "GR15": R(recipe="content", pointer="dossier 17: v3_overlap {0:180, 1:148} + folders", hand=True),
    "GR16": R(recipe="health", kind="hubs"),
    "GR17": R(recipe="boundary", a=11, b=7),
    "GR18": R(recipe="boundary", a=9, b=6, hand=True),
    "GR19": R(recipe="onboarding", sub=6, hand=True),
    "GR20": R(recipe="flow", rx="CascadeDelete", hops=1, hand=True),
    "GR21": R(recipe="trophic_inversion", sub=11),
    "GR22": R(recipe="health", kind="events"),
    "GR23": R(recipe="health", kind="config_reach"),
    "GR24": R(recipe="health", kind="bridges"),
    "GR25": R(dup_of="M01"),
    "GR26": R(recipe="content", pointer="dossier 11: v3_overlap {6:78, 7:72}", hand=True),
    "GR27": R(recipe="impact", name="UserSocialConnectionRepository.java"),
    "GR28": R(dup_of="M07"),
    "GR29": R(recipe="health", kind="top_indegree"),
    "GR30": R(recipe="impact", name="CorsProperties.java"),
    "GR31": R(recipe="cohort", metapath="P_R_P"),
    "GR32": R(recipe="content", pointer="e2e-tests fixtures mirror BE contracts by name; no FE->BE edges modeled (separate repos)", hand=True),
    "GR33": R(dup_of="M08"),
}

# Hand-authored insight prefixes (verified against executed rows on second run).
HAND = {
    "BE01": "PaymentsDisabledBootGuard (sub-11, Rule) refuses startup when payments are flag-disabled but the subscription table holds a state the pipeline cannot reconcile — the documented prod crash-loop mode.",
    "BE06": "It returns 403 for any authenticated request whose user lacks the currently required consents — the GDPR gate, running after auth and before controllers.",
    "BE12": "Backend messages and transactional emails localize via messages_en/pl.properties; AppLanguageFilter resolves and stores the request language.",
    "BE13": "Day-1: native PostgreSQL 16 service (NOT Docker — both would own checkitout_local_db, split-brain risk), then the doc's steps; Liquibase migrates on boot.",
    "BE14": "11 findings. Fixed on greenfield: SSRF StorageUrlValidator (3.1), IDOR getUserByUserId (3.7), Vimeo allowlist (3.5), SecureRandom ticket refs + IP rate-limit (3.3/3.2), nginx single-source headers (3.10/3.11). Accepted by design: flat-admin (3.6/3.9). Deferred: Angular 22 (3.4), nonce CSP (3.8).",
    "BE15": "The strategy doc defines when unreferenced uploads are swept; the cron layer executes it against attachment references.",
    "BE16": "Upgrade path: DTO -> SubscriptionService validates the plan transition -> StripeService updates the Stripe subscription -> webhook confirms -> SubscriptionEvent recorded, BillingPeriod rolled, campaign quota re-derived from the new plan.",
    "BE17": "Invoice failure -> InvoiceRetryService marks the record -> InvoiceRetryCronJob replays pending invoices on schedule until success or terminal give-up; InvoiceStatus tracks the lag.",
    "BE20": "All 9 crons share the ShedLock template (ConsentEnforcementCronJob is the reference): a DB-held named lock prevents double execution across instances; schedules live in @Scheduled/@SchedulerLock annotations (content).",
    "BE21": "Publishing a TermsVersion opens a grace window; TermsGraceProcessorCronJob advances state as it expires; ConsentEnforcementFilter then blocks non-re-consented users, and the FE re-consent modal is driven by the response headers.",
    "BE22": "The application-no-redis.yml profile exists precisely for this: cache lookups (UserCacheService) fall back to Postgres with higher latency, and rate limiting loses cross-instance state.",
    "BE23": "Admin ban flips account state; BannedUserAuthorizationFilter checks it on every request and returns 403 even with valid Firebase auth — enforcement is at the filter, not in controllers.",
    "BE24": "CampaignLimitService reads the company's plan (CompanySubscription + BillingPeriod) and blocks PartnershipOpportunity creation past quota — the same 3-hop path M08 measured.",
    "BE25": "Right-to-erasure: the request lands in PendingDataDeletionRequestRepository; after the configured deferral window DeferredDeletionCronJob executes via UserAccountOrchestrator into cascade delete.",
    "BE27": "The house pattern: the status change publishes an event via @TransactionalEventListener after commit; NotificationEventListener materializes the user-visible notification — notification failure can never roll back the business change.",
    "BE28": "A cron has no request context, so language comes from the recipient's stored preference, not AppLanguageFilter; EmailCronJob renders from messages_{lang}.properties per user.",
    "BE29": "resources/ is not indexed by the scan (by design). From the repo: service-account*.json (Firebase) and keystore.p12 in src/main/resources — both on the public-release scrub checklist.",
    "BE30": "Same mechanism as BE01 seen from the symptom side: a restored prod dump carries a non-FREE_ACTIVE subscription row; with payments flag-disabled, the boot guard refuses to start — reconcile the row (documented outage mode).",
    "BE31": "The company changed nothing: a billing-period rollover or downgrade cron re-derived the quota, and CampaignLimitExceededException now fires — check BillingPeriod transitions on the day it started.",
    "BE34": "The charge is synchronous, invoicing is asynchronous-with-retry: InvoiceRetryCronJob replays failed Fakturownia calls on its cadence, so the invoice can trail the payment by cadence x retries.",
    "BE35": "Every entity carries @Version (optimistic locking): the second tab saves with a stale version and gets a conflict instead of silently overwriting — by design; reload and retry.",
    "BE36": "RequestCorrelationFilter stamps a correlation id per request and propagates it via MDC — one user action, one id, across every log line; search logs by that id.",
    "BE37": "StorageUrlValidator (the pentest 3.1 SSRF fix, 2026-09-02) post-dates the scan — delta-scan item. Behaviour: attachment URLs validate against a storage-host allowlist; other hosts are rejected regardless of the file.",
    "FE03": "Add the key to BOTH src/assets/i18n/en.json and pl.json (check:i18n-parity fails on drift), reference it via Transloco; the i18n version buster invalidates cached bundles.",
    "FE04": "Two locales, en and pl; default is pl (Transloco config in app.config.ts).",
    "FE05": "A deliberate pin (iteration 107): browser-language auto-pick broke SSR hydration — server rendered pl, browser re-rendered en, mismatch/flicker — so first load is pl and switching is explicit; the app.config.ts comment records exactly this.",
    "FE06": "error.interceptor catches 401 globally -> SessionState.clear() -> router navigates to /auth/sign-in; no per-component handling anywhere.",
    "FE10": "There is no stored token by design: the session is a cookie; ApiConfiguration sets withCredentials so it rides every call — hence deliberately no Authorization-header interceptor. (The client layer is the collapsed 'api' node in the graph.)",
    "FE11": "basePath /api on ApiConfiguration; proxy.conf.js forwards /api -> localhost:8080 in dev. (Both live inside the collapsed 'api' node — the 150-edge articulation point.)",
    "FE12": "npm run openapi:gen regenerates src/app/api from the BE spec; openapi:cycle runs the BE-regen + FE-regen loop. Never hand-edit generated files.",
    "FE14": "Public without login: landing, technical-survey and its chapters, team, grants, support ticket create/status — aggregated from the guards in app.routes.ts.",
    "FE16": "Kept as redirects: /subscription/success and /cancel (Stripe return URLs), /plan-billing, /welcome, /ui-component-samples -> __sandbox — old bookmarks and Stripe-configured returns survive the rewrite.",
    "FE18": "SeoTitleStrategy renders 'translated title | site' from route title keys and re-titles live on language switch. (The strategy file is absent from the scan — delta item.)",
    "FE19": "e2e-tests/visual-parity: component-pairs.ts maps legacy<->greenfield components, parity:capture builds baselines, pixel-diff flags divergence. (e2e-tests only partially indexed — gap.)",
    "FE20": "Projects: desktop chromium, Pixel 7, iPhone 14, iPad Pro 11; the Safari ones require npx playwright install webkit first.",
    "FE22": "The integration tier's _trace framework records both frontends' API call sequences, canonicalizes (order, volatile fields), and diffs — trace-equivalence catches drift codegen cannot. (Partially indexed — gap.)",
    "FE23": "_framework/real-login.ts logs in with real credentials from e2e-tests/.env; @login-real specs self-skip when secrets are absent. (Not indexed — gap.)",
    "FE26": "ssr-cookie-forward.interceptor (server-only) forwards the browser's session cookie onto SSR-side API calls, so the auth guard's server probe sees the session. Added after the scan — delta item.",
    "FE27": "provideClientHydration(withEventReplay()): clicks before hydration are captured and replayed once listeners attach — early interactions are not lost.",
    "FE28": "shell-headers.interceptor taps X-Consent-Required and email-verification headers on every response and feeds shell-status.service, which drives the shell banners.",
    "FE30": "check:full = check:no-legacy-ui + check:api-wrappers + check:i18n-parity + check:visual-fixture-coverage + lint-staged + typecheck + build:check (ng build, strictTemplates) + jest — the pre-commit floor, ~17s clean.",
    "FE33": "The spec stopped exposing some payment/2FA models, so their types are hand-frozen in core/api-frozen (subscription.client.ts + hidden-models.ts). Never delete src/app/api or the frozen client. (api-frozen not indexed — gap.)",
    "GR15": "Dossier evidence: v3_overlap {0:180, 1:148} shows two latent halves; the natural cut is the old v3 boundary refined by cohort-fiber cuts and top-level folders (feature/ vs core+layout). This is Curator decision material (C2), with the fan-out budget forcing the split.",
    "GR18": "The auth estate is three subsystems: sub-9 (auth/feature rules), sub-6 (TOTP/step-up + user cache), sub-8 (recaptcha/specification rules). The 9<->6 seam is the working coupling; sub-8 attaches through validation rules.",
    "GR19": "The TOTP/step-up subsystem is a service layer, not self-starting: external in-edges land on UserCacheService, FirestoreService and TotpFirestoreService, and it has no actor roots — it is invoked from sub-9's controllers.",
    "GR20": "The cascade-deletion subsystem fans out over 21 files in one hop (6 MODIFIES, 6 USES against repositories); the deletion ORDER is encoded in AdminCascadeDeleteServiceImpl — content, not structure: the graph gives the touch set, the class gives the sequence.",
    "GR26": "Dossier 11 v3_overlap {6:78, 7:72}: two v3 subsystems (subscriptions; consent) merged into one v4 subsystem — consolidation, not separation. The curation call is whether to keep the merge; the seam data says yes: consent enforcement is billing-adjacent by construction.",
    "GR32": "Not modeled, honestly: FE and BE are separate repos with no cross-repo edges, so fixture-to-backend mapping is nominal (MSW fixtures mirror BE contracts by name). A cross-repo contract layer is a possible C4 enrichment.",
}


def _component_labels(deg, A):
    """Weakly-connected component id per vertex (undirected support of A). deg==0 -> own id."""
    n = len(deg)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = (A + A.T).nonzero()
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def _gauge_per_component(h, deg, comp):
    """Normalise MacKay heights per weakly-connected component (min over deg>0 nodes = 0).

    GAUGE CHANGE (2026-09-02, task 63) — the one semantic change sanctioned in this file.
    Trophic height solves L h = d_in - d_out, and L is singular with nullity equal to the
    number of weakly-connected components, so h is determined only up to an INDEPENDENT
    additive constant PER COMPONENT. The previous single global shift (`h -= h[deg>0].min()`)
    imposed one origin on components that share no edges; worse, np.linalg.lstsq returns the
    minimum-norm solution, so each component's constant was assigned by the pseudoinverse —
    an artifact, not a measurement, and unstable under changes to unrelated components.
    Symptom that exposed it: c1_dossiers.py already gauges per component, and the two
    disagreed on trophic_span for every multi-component subsystem, always c3 >= c1.

    Deliberately DUPLICATED (not imported) in graph/scripts/c3_organisation.py mackay_heights.
    The two remain independent implementations so the conformance differential can still catch
    a divergence between them; a shared helper would make that gate partly self-referential.
    """
    groups = {}
    for i in range(len(h)):
        if deg[i] > 0:
            groups.setdefault(comp[i], []).append(i)
    for members in groups.values():
        base = min(h[i] for i in members)
        for i in members:
            h[i] -= base
    return h


def r_trophic_inversion(g, sub):
    import numpy as np
    # VERTEX IDENTITY is the node id, never the file name — see the Graph.__init__ note.
    ids = sorted(n["nid"] for n in g.nodes if n["sub"] == sub)  # determinism law
    name_of = {n["nid"]: n["name"] for n in g.nodes}
    idx = {n: i for i, n in enumerate(ids)}
    A = np.zeros((len(ids), len(ids)))
    for e in g.edges:
        if e["asub"] == sub and e["bsub"] == sub:
            A[idx[e["aid"]], idx[e["bid"]]] += 1
    din, dout = A.sum(0), A.sum(1)
    deg = din + dout
    L = np.diag(deg) - A - A.T
    h = np.linalg.lstsq(L, din - dout, rcond=None)[0]
    comp = _component_labels(deg, A)
    h = _gauge_per_component(h, deg, comp)   # was: single global shift
    # COMPONENT-LOCAL COMPARISON (2026-09-02, task 63 parts b/c). Heights are gauged per
    # weakly-connected component, so a median pooled over the whole subsystem compares numbers
    # that do not share an origin. Each controller is therefore tested against the median of
    # ITS OWN component. A component holding no Service has no in-component median, so the
    # test is UNDEFINED there and we report it as such rather than borrowing a foreign median
    # — which is what the pooled version silently did (measured: sub-7's only reported
    # inversion was a controller compared against a median from a different component).
    # Edges point controller -> service, so controllers sit LOW (upstream) when the flow
    # order is healthy; an inversion is a controller ABOVE its own component's service median.
    svc_by_c, ctrl = defaultdict(list), []
    for i, nid in enumerate(ids):
        if deg[i] <= 0:
            continue
        n = name_of[nid]
        if "Service" in n:
            svc_by_c[comp[i]].append(h[i])
        if "Controller" in n:
            ctrl.append((n, h[i], comp[i]))
    med_by_c = {c: sorted(v)[len(v) // 2] for c, v in svc_by_c.items()}
    inv = [(n, round(v, 2)) for n, v, c in ctrl if c in med_by_c and v > med_by_c[c]]
    und = [(n, round(v, 2)) for n, v, c in ctrl if c not in med_by_c]
    rows = ([[n, f"{v:.2f}", "CONTROLLER_ABOVE_SERVICE_MEDIAN"] for n, v in inv]
            + [[n, f"{v:.2f}", "UNDEFINED_NO_SERVICE_IN_COMPONENT"] for n, v in und]
            ) or [["none", "", ""]]
    meds = ", ".join(f"c{ci}:{m:.2f}" for ci, m in enumerate(
        m for _, m in sorted(med_by_c.items())))
    ans = (f"Sub-{sub} trophic check (MacKay heights on the internal digraph, gauged PER "
           f"weakly-connected component; controllers are upstream/low when healthy). Compared "
           f"within each component — service medians {meds or 'none'}; inversions (controller "
           f"above its OWN component's median): "
           f"{', '.join(n for n, _ in inv) or 'NONE — the controller->service flow order holds throughout'}"
           f"; controllers in Service-less components (test undefined): "
           f"{', '.join(n for n, _ in und) or 'none'}.")
    return rows, ans, False


def load_bank():
    out = {}
    for pfx, fn in (("BE", "qminer_be.jsonl"), ("FE", "qminer_fe.jsonl"), ("GR", "qminer_graph.jsonl")):
        with open(os.path.join(QDIR, "raw", fn), encoding="utf-8") as f:
            for i, line in enumerate(l for l in f if l.strip()):
                out[f"{pfx}{i+1:02d}"] = json.loads(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ladybug", "neo4j"], default="ladybug")
    args = ap.parse_args()
    bank = load_bank()
    exemplars = {json.loads(l)["id"]: json.loads(l)
                 for l in open(os.path.join(QDIR, "mfq_gold.jsonl"), encoding="utf-8")}
    if args.backend == "ladybug":
        from ladybug_store import Store
        g = LadybugGraph(Store(read_only=True))
    else:  # migration-differential only
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver(BOLT, auth=AUTH)
        with drv.session() as s:
            g = Graph(s)
        drv.close()

    recs, digest = [], ["# Q1 gold FULL — digest (auto answers shown; HAND prefixes applied where present)\n"]
    for qid, q in bank.items():
        m = MANIFEST[qid]
        rec = dict(id=qid, q=q["q"], aliases=q["aliases"], role=q["role"], stratum=q["stratum"],
                   why=q.get("why"), worth=2)
        if m.get("dup_of") and m["dup_of"] in exemplars:
            e = exemplars[m["dup_of"]]
            rec.update(dup_of=m["dup_of"], archetype=e["archetype"], gold_status=e["gold_status"],
                       gold_fingerprint=e["gold_fingerprint"], gold_answer=e["gold_answer"],
                       depends_on_subsystems=e["depends_on_subsystems"], worth=3)
            recs.append(rec)
            continue
        recipe = m["recipe"]
        empty = False
        if recipe == "locate":
            rows, auto, empty = r_locate(g, m["rx"], m.get("note", ""))
        elif recipe == "impact":
            rows, auto, empty = r_impact(g, m["name"])
        elif recipe == "flow":
            rows, auto, empty = r_flow(g, m["rx"], m.get("hops", 2))
        elif recipe == "boundary":
            rows, auto, empty = r_boundary(g, m["a"], m["b"])
        elif recipe == "cohort":
            rows, auto, empty = r_cohort(g, m.get("name"), m.get("metapath"))
        elif recipe == "health":
            rows, auto, empty = r_health(g, m["kind"])
        elif recipe == "onboarding":
            rows, auto, empty = r_onboarding(g, m["sub"])
        elif recipe == "trophic_inversion":
            rows, auto, empty = r_trophic_inversion(g, m["sub"])
        elif recipe == "content":
            rows, auto = [], f"Answer lives in content, not structure. Pointer: {m['pointer']}."
        rows, fp = canon(rows)
        status = ("COVERAGE_GAP" if (empty and recipe == "locate")
                  else "CONTENT_POINTER" if recipe == "content"
                  else "EXECUTED")
        hand = HAND.get(qid, "")
        answer = (hand + (" " if hand and auto else "") + auto).strip()
        subs = sorted({int(c) for r in rows for c in r
                       if isinstance(c, int) or (isinstance(c, str) and c.isdigit() and int(c) <= 18)}) if rows else []
        rec.update(archetype=recipe, recipe_params={k: v for k, v in m.items() if k not in ("recipe", "hand", "dup_of")},
                   gold_result_excerpt=rows[:20], gold_result_count=len(rows),
                   gold_fingerprint=fp, gold_status=status,
                   depends_on_subsystems=subs, gold_answer=answer,
                   needs_hand=bool(m.get("hand")) and not hand)
        recs.append(rec)
        digest.append(f"## {qid} [{status}]{' NEEDS_HAND' if rec['needs_hand'] else ''} — {q['q']}\n"
                      f"rows={len(rows)} fp={fp}\nAUTO: {auto}\n" +
                      ("```\n" + "\n".join(" | ".join(map(str, r)) for r in rows[:10]) + "\n```\n" if rows else ""))
    with open(os.path.join(QDIR, "mfq_all.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(QDIR, "GOLD_ALL_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(digest))
    nh = [r["id"] for r in recs if r.get("needs_hand")]
    st = Counter(r["gold_status"] for r in recs)
    print(f"OK {len(recs)} records. status={dict(st)}. needs_hand({len(nh)}): {','.join(nh)}")


if __name__ == "__main__":
    main()
