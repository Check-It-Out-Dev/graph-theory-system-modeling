# CodeMap GrothendieckV5 P4 — the curation write for CheckItOutV3.
#
# Judgement layer only. It NEVER touches a measured property (sub, embeddings,
# hyperedges, assignment_*). Curated membership lives in Member edges plus a NEW
# `curated_sub` column written ONLY on moved/split nodes.
#
# Ledger law applied here:
#   L1 — the 19 Nav rows already exist with role CANDIDATE / name_status PROVISIONAL.
#        They are UPDATED IN PLACE (identity + Member edges preserved), never replaced
#        by parallel L2 nodes; MFQ depends_on_subsystems stamps key on sub_id.
#   L3 — the repo-purity invariant holds; the 9 sub-17 children are all FE, the rest all BE.
#   Supersession is never erasure: merged navigators keep their row, lose their Guides
#        edge, and gain a SupersededBy edge to the absorbing navigator.
#
# STORE PORT (2026-09-02): Neo4j -> Ladybug Store. Semantics preserved 1:1 with the
# executed 2026-09-02 batch, with three deliberate translations:
#   - membership is keyed on nid, not name (ledger L11 — duplicate basenames exist);
#   - long-tail annotations (name_status, curated_*, dissents, verdicts) live in the
#     `props` JSON column via Store.merge_props; curated_subsystem -> `curated_sub` col;
#   - the (V3Master)-[:HAS_DECISION]->(CurationDecision) edge is DROPPED: decisions are
#     standalone rows whose `batch` column carries the linkage (no Master-decision rel
#     table in the store schema, by design).
#
# Usage: PYTHONUTF8=1 python c2_curation.py [--dry]

import argparse
import datetime as _dt
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "authoring")))
from ladybug_store import Store

FE = "C:/Users/Norbert/IdeaProjects/checkItOut-fe-greenfield/"
CLUE_VERSION, BATCH, BY = "curated-v1", "2026-09-02-curation", "GrothendieckV5"

# ---------------------------------------------------------------- sub-17 split scheme
# Vertical (domain) parts. Chosen over the horizontal core/feature/fixture scheme by
# measurement: modularity on the sub-17 subgraph Q=0.604 vs 0.120, aggregate cohesion
# 0.489 vs 0.241. The horizontal scheme cuts the single largest internal seam in the
# subsystem (core/auth <-> feature/auth, 40 edges).
PARTS = {
    170: ("FE auth, 2FA & interceptors",
          ["C:auth", "F:auth", "C:step-up", "C:two-factor", "C:interceptors"]),
    171: ("FE opportunities & applications",
          ["F:opportunities", "C:opportunities", "F:applied-opportunities",
           "C:applied-opportunities", "F:collaborations", "F:grants"]),
    172: ("FE onboarding survey", ["F:survey"]),
    173: ("FE fixtures & E2E harness", ["sandbox", "e2e"]),
    174: ("FE plan, billing & consent",
          ["F:plan-billing", "C:subscription", "C:api-frozen", "C:config",
           "C:consent", "C:legal", "F:legal"]),
    175: ("FE support & help centre", ["F:support", "C:support"]),
    176: ("FE profile, settings & admin",
          ["F:profile", "C:user", "C:upload", "F:settings", "C:social", "C:preferences",
           "F:preferences", "F:addresses", "C:address", "F:company", "C:registry",
           "F:team", "F:admin", "C:admin", "C:dictionary"]),
    177: ("FE shell, i18n & generated client",
          ["layout", "C:shell", "F:shell", "C:theme", "C:notifications", "C:i18n",
           "shared", "C:health", "F:error-page", "C:rate-limit", "F:landing", "root"]),
    178: ("FE demo mode", ["C:demo", "F:demo"]),
}
# 173 is a fixture/data layer (85% Resource purity, 4 internal edges, uniformly consumed),
# not a feature slice — the manual forbids silently keeping such a set typed as a slice.
CHILD_ROLE = {173: "LAYER"}

# ------------------------------------------------------- curated names / roles for L2
# Naming law: top_terms + dominant_layer + medoids + folders together; 2-4 words; what it DOES.
CURATED = {
    0:  ("User preferences & geo distance", "SLICE"),
    1:  ("Address resolution & storage", "SLICE"),
    2:  ("Social connection data model", "LAYER"),
    3:  ("Rate limits & runtime config", "SLICE"),
    4:  ("Partnership opportunity lifecycle", "SLICE"),
    5:  ("Account deletion & Instagram sync", "SLICE"),
    6:  ("Two-factor auth & user cache", "SLICE"),
    7:  ("Translatable exceptions & logging", "LAYER"),
    8:  ("Validation, crypto & query specs", "SLICE"),
    9:  ("Auth journeys & BDD harness", "SLICE"),
    10: ("User identity & token exchange", "SLICE"),
    11: ("Subscriptions, payments & consent", "SLICE"),
    12: ("Notifications & domain events", "SLICE"),
    14: ("FAQ content & categories", "SLICE"),
    15: ("Support tickets & attachments", "SLICE"),
    16: ("Status enums & OpenAPI contract", "LAYER"),
    17: ("Greenfield Angular frontend", "GROUP"),
}
MERGED = {13: (4, "Partnership opportunity lifecycle"), 18: (177, "FE shell, i18n & generated client")}


def bucket(p):
    r = p.replace(FE, "")
    if r.startswith("e2e-tests/"):
        return "e2e"
    if r.startswith("src/app/sandbox"):
        return "sandbox"
    if r.startswith("src/app/feature/"):
        return "F:" + r[16:].split("/")[0]
    if r.startswith("src/app/core/"):
        return "C:" + r[13:].split("/")[0]
    if r.startswith("src/app/shared"):
        return "shared"
    if r.startswith("src/app/layout"):
        return "layout"
    return "root"


# ------------------------------------------------------------------ decision log (P4)
# rationale MUST cite dossier fields / measured numbers.
DECISIONS = [
    dict(subsystem=17, action="SPLIT+RETYPE+RENAME", target="170-178 (9 children), role GROUP",
         rationale="MEGA trigger tripped: share 28.62% > 20% (dossier size 405/1415). Split "
         "vertically by domain, not horizontally by layer, because the horizontal scheme cuts "
         "the largest internal seam in the subsystem (core/auth<->feature/auth, 40 edges) — the "
         "same core/feature pairing repeats for demo(10), opportunities(10), support(10), "
         "user<->profile(9), subscription<->plan-billing(8). Parent kept as a GROUP rather than "
         "flat-replaced: sub-17's cohesion 0.723 is the highest of any candidate, and ledger L1 "
         "requires preserving sub_id identity for MFQ depends_on_subsystems stamps.",
         evidence="modularity on sub-17 subgraph: vertical Q=0.604 vs horizontal Q=0.120 (5.0x); "
         "aggregate cohesion 0.489 vs 0.241; largest child share 5.16% of corpus (was 28.62%); "
         "fan-out 9 = budget; cohort-parent corroboration — the v3 parent cut concentrates >=84% "
         "in 6 of 9 parts (AUTH 62/69 v3=0, OPPS 36/43 v3=12, SURVEY 37/42 v3=1, SUPPORT 20/20 "
         "v3=1, BILLING 20/23 v3=0, DEMO 14/16 v3=1); per-child cohesion SURVEY 1.000, DEMO 0.824, "
         "AUTH 0.559, SUPPORT 0.529, OPPS 0.524, SHELL 0.521, BILLING 0.467, PROFILE 0.455, "
         "SANDBOX 0.065 (retyped LAYER for that reason). Louvain on the same subgraph reaches "
         "Q=0.815 only by shattering into 182 communities (178 of them <3 nodes) — unusable "
         "against the fan-out<=9 budget; the vertical scheme keeps 74% of that Q with 9 parts."),
    dict(subsystem=18, action="MERGE", target="17 -> child 177 (FE shell, i18n & generated client)",
         rationale="n=1 < 3 (MICRO flag) and the strongest dominant-seam-partner measurement in "
         "the whole graph: 100.0% of its 156 external edges go to sub-17, direction 100% fan-in. "
         "Placed in the platform child rather than a feature child because all 9 children consume "
         "it with no dominant consumer (max child share 23.7%).",
         evidence="sub-18 external=156, sub-17:156 (100.0%); in=156 out=0; consumers=1; "
         "consumption by child: SANDBOX 37, PROFILE 33, OPPS 29, AUTH 20, BILLING 14, SUPPORT 12, "
         "SHELL 9, DEMO 2 — max 23.7%, no dominant child. Merge internalises all 156 edges."),
    dict(subsystem=13, action="MERGE", target="4 (Partnership opportunity lifecycle)",
         rationale="RETYPE to role LAYER was CONSIDERED AND REFUTED. A layer is a supplier "
         "consumed by many; sub-13 is the exact opposite — a consumer satellite. Direction profile "
         "in=9 out=361 (in-share 2.4%) with only 2 distinct consumers, against sub-16's 93.7%/12 "
         "and sub-7's 91.1%/15. The 90% Rule purity that raised the LAYER flag is real but it is "
         "the purity of one slice's test suite, not of a cross-cutting layer. The manual's own "
         "trigger offers 'RETYPE or dissolve into served slices' — the served slice is singular, "
         "so it dissolves.",
         evidence="dossier: purity Rule 90% (44/49), ext_ratio 0.938, seam IMPORTS->sub-4 n=188 "
         "(223 undirected, 60.3% of its 370 external edges), v3 overlap {4:48, 7:1} = 48/49 in "
         "the same parent cell as sub-4. MEASURED DELTA: sub-4 cohesion 0.362 -> 0.478 (+32%), "
         "internal edges 515 -> 761, size 123 -> 172 = 12.16% of corpus, still under the 20% "
         "MEGA trigger. DISSENT PRESERVED: the LAYER reading is recorded on the node as "
         "curation_dissent."),
    dict(subsystem=16, action="RETYPE+RENAME", target="role LAYER",
         rationale="MERGE was CONSIDERED AND REFUTED. The manual's merge trigger requires "
         "ext_ratio > 0.9 AND one dominant seam partner; the first holds (1.000) and the second "
         "fails — the top three partners are tied at 8 edges each (12.7%). The direction profile "
         "is a textbook shared-kernel signature instead: 93.7% fan-in, 12 distinct consumers, zero "
         "internal edges. These are the status/compensation enums, their i18n message bundles and "
         "the OpenAPI spec config — shared vocabulary, not a fragment of anybody.",
         evidence="ext_ratio 1.000, internal edges 0, external 63 spread over 12 consumers with "
         "max partner 12.7% (sub-6/9/11 tied at 8); in=59 out=4; entry AccountStatus.java with 51 "
         "in-edges. EVERY merge candidate LOWERS the host's cohesion: ->sub-4 0.362->0.353, "
         "->sub-6 0.177->0.173, ->sub-9 0.242->0.234, ->sub-11 0.489->0.471. NOTE: this contradicts "
         "the manual's purity>0.85 layer trigger (measured purity is Resource 67%) — the fan-in "
         "signature is the direct evidence and the measurement wins."),
    dict(subsystem=2, action="RETYPE+RENAME", target="role LAYER",
         rationale="MERGE was CONSIDERED AND REFUTED. ext_ratio 0.993 trips the first half of the "
         "trigger but no partner dominates: sub-4 33.1% vs sub-5 24.4%, a ratio of 1.36. The "
         "direction profile says supplier, not fragment: 86.2% fan-in across 8 distinct consumers. "
         "This is the UserSocialConnection/Platform data model with its mapper and enum "
         "translation — merging it into either candidate would bury an 8-consumer kernel inside "
         "one consumer for a negligible gain.",
         evidence="ext_ratio 0.993, internal edges 1, external 160: sub-4 53 (33.1%), sub-5 39 "
         "(24.4%), sub-6 21 (13.1%), sub-13 13, sub-10 11; in=138 out=22 (86.2% fan-in), "
         "consumers=8; entry UserSocialConnectionRepository 40, UserSocialConnection 30, "
         "PlatformRepository 29. MEASURED DELTA of the rejected merges: +2 -> sub-4 cohesion "
         "0.362->0.372 (+0.010); +2 -> sub-5 0.305->0.311 (+0.006). Both negligible. Corroborated "
         "from the FE side: the two flagged FE social-connection files carry "
         "assignment_crossrepo_affinity=2, i.e. the unconstrained kNN names sub-2 as a cross-repo "
         "domain seam — the behaviour of a shared model, not of a fragment."),
    dict(subsystem=7, action="RETYPE+RENAME", target="role LAYER",
         rationale="Not in the curation queue, retyped for consistency: the same layer criterion "
         "that promoted sub-2 and sub-16 selects sub-7 more strongly than either, and applying a "
         "criterion to some candidates but not all would make the taxonomy arbitrary. This is the "
         "exception hierarchy and translatable-message infrastructure the whole backend imports.",
         evidence="in=422 out=41 (91.1% fan-in), 15 distinct consumers, max partner 13.4% — the "
         "strongest layer signature in the graph. ext_ratio 0.911. Entry points are pure supply: "
         "ResourceNotFoundException 101 in-edges, ValidationTranslatableException 96, "
         "InsufficientPermissionsException 58. No membership change."),
    dict(subsystem=8, action="KEEP+RENAME", target="—",
         rationale="MERGE-CHECK resolved as KEEP. ext_ratio 0.944 trips the first half of the "
         "trigger, but no partner dominates (sub-4 26.8%) and the direction profile is balanced "
         "(49.8% fan-in), so it is neither a fragment of one neighbour nor a supplier layer. It "
         "is a genuine shared-utility slice: query specification building, HMAC/TOTP crypto, "
         "validation annotations and recaptcha, with their unit tests.",
         evidence="ext_ratio 0.944, external 209: sub-4 56 (26.8%), sub-6 26 (12.4%), sub-9 25 "
         "(12.0%), sub-7 21 (10.0%) — a spread, not a seam; in=104 out=105; consumers=12; entry "
         "SpecificationBuilder 48 in-edges, HmacUtils 11, TotpCodeGenerator 10. MEASURED DELTA of "
         "the rejected merge: 8+4 cohesion 0.362 -> 0.368 (+0.006), negligible. n=49 (3.5%) is far "
         "under the SPLIT trigger, so the internal utils/tests heterogeneity is left alone."),
]
for sid in [0, 1, 3, 4, 5, 6, 9, 10, 11, 12, 14, 15]:
    DECISIONS.append(dict(
        subsystem=sid, action="KEEP+RENAME", target="—",
        rationale="No trigger tripped. Renamed per the naming law (top_terms + dominant_layer + "
                  "medoids + folders together) to state what the subsystem DOES; name_status "
                  "PROVISIONAL -> CURATED." + (
                      " Absorbs sub-13 (see that decision); size 123 -> 172."
                      if sid == 4 else ""),
        evidence="see dossier subsystem_%d.json: size/purity/ext_ratio/top_terms/medoids/entry" % sid))
DECISIONS.append(dict(
    subsystem=-1, action="ASSIGNMENT-REVIEW", target="14 flagged delta assignments",
    rationale="All 14 assignment_flagged nodes CONFIRMED, none moved. The 8 low-margin BE "
    "assignments are each corroborated by folder evidence that the margin alone could not see; "
    "the 6 FE assignments were flagged only because the unconstrained kNN dissented across the "
    "repo boundary, and ledger L3 makes the repo-purity constraint (precision 1.0 at n=1374) "
    "binding. Those 6 dissents are now ACTIONABLE rather than merely preserved: the split places "
    "each in a child, and 5 of 7 cross-repo affinities land in the child whose BE counterpart the "
    "unconstrained kNN named — an independent cross-validation of the split.",
    evidence="CONFIRMED, margin<0.6 group: VimeoUrls(0.25)/SocialPostUrl(0.333)/"
    "VimeoUrlsValidator(0.333)/SocialPostUrlValidator(0.5) -> sub-8, all four in "
    "platform/common/validation/ alongside sub-8's own entry point ValidationPatterns.java; "
    "TicketAccessTokenService(0.25) -> sub-15, in platform/support/ticket/services/ matching "
    "sub-15 terms ticket/support/attachment; logback-spring.xml(0.583) -> sub-3 matching terms "
    "yml/application. CONFIRMED, margin>=0.6: DevLiteUploadController(0.833) -> sub-3 (dev-lite "
    "config surface). CONFIRMED, cross-repo dissent group (all margin 0.917-1.0, all FE): "
    "subscription.client.ts xrepo=11 -> child 174 BILLING (BE sub-11 IS subscriptions/payments/"
    "consent — exact match); public-config.service.ts xrepo=11 -> 174 (match); "
    "social-connections.service.ts xrepo=2 -> 176 PROFILE and social-connections-settings."
    "component.ts xrepo=2 -> 176 (BE sub-2 IS the social connection model — match); "
    "rate-limit-state.service.ts xrepo=3 -> 177 SHELL (BE sub-3 IS rate limits/runtime config — "
    "match); recorder.ts xrepo=7 -> 173 SANDBOX (partial: BE sub-7 is the logging layer, the "
    "trace recorder mirrors it); real-login.ts xrepo=9 -> 173 SANDBOX (DISSENT RETAINED: BE sub-9 "
    "is auth journeys, so child 170 AUTH is arguable; kept with the E2E harness on folder "
    "evidence, recorded as curation_dissent on the node)."))

# Navigator fields snapshotted before an in-place update (phase 1) or a merge (phase 4).
SNAP_COLS = ["clue_version", "name", "role", "ai_summary", "responsibilities", "caveats",
             "entry_points", "spines", "contracts", "size", "external_ratio",
             "dossier_fingerprint"]
SNAP_PROPS = ["name_status", "layer_profile", "generated_at"]


def jdump(v):
    return json.dumps(v, ensure_ascii=False, default=str)


def nav_row(s, sid):
    cols = ", ".join(f"n.{c} AS {c}" for c in SNAP_COLS)
    r = s.one(f"MATCH (n:Nav) WHERE n.sub_id = $sid RETURN {cols}, n.props AS props",
              dict(sid=sid))
    if r is None:
        return None
    p = json.loads(r.pop("props") or "{}")
    for k in SNAP_PROPS:
        r[k] = p.get(k)
    return r


def snap_body(row, reason):
    body = {k: row.get(k) for k in SNAP_COLS + SNAP_PROPS}
    body.update(level="L2", t_valid=row.get("generated_at"), reason=reason)
    return body


def main(dry):
    s = Store(read_only=dry)
    now = _dt.datetime.now().isoformat()

    nodes = s.q("MATCH (n:Entity) RETURN n.nid AS nid, n.name AS name, "
                "n.file_path AS path, n.sub AS sub")
    nodes.sort(key=lambda n: n["nid"])               # determinism at the boundary
    assign = {}          # nid -> curated child sub_id (sub-17 members + the sub-18 node)
    name_of = {}
    for n in nodes:
        name_of[n["nid"]] = n["name"]
        if n["sub"] == 17:
            b = bucket(n["path"])
            pid = next((k for k, (_, bs) in PARTS.items() if b in bs), None)
            if pid is None:
                sys.exit(f"FATAL: bucket {b} unassigned ({n['name']})")
            assign[n["nid"]] = pid
        elif n["sub"] == 18:
            assign[n["nid"]] = 177
    sizes = Counter(assign.values())
    print("child sizes:", dict(sorted(sizes.items())), "total", sum(sizes.values()))
    if dry:
        print(f"DRY — no writes. Would write: {len(CURATED)} navigator supersessions, "
              f"{len(PARTS)} children, {len(assign)} Member edges + curated_sub, "
              f"{len(MERGED)} merges, {len(DECISIONS)} CurationDecision rows")
        return

    # 1) snapshot + update in place every surviving navigator (L1: identity preserved)
    for sid, (name, role) in sorted(CURATED.items()):
        row = nav_row(s, sid)
        s.create("ClueSnap", dict(
            snap_id=f"c2-{BATCH}-{sid}", sub_id=sid, taken_at=now, t_created=now,
            t_expired="", superseded_by=BATCH,
            body=jdump(snap_body(row, f"curation batch {BATCH}: provisional candidate "
                                      f"judged into a curated subsystem"))), "snap_id")
        s.conn.execute(
            "MATCH (n:Nav) WHERE n.sub_id = $sid "
            "SET n.name = $name, n.role = $role, n.clue_version = $cv",
            parameters=dict(sid=sid, name=name, role=role, cv=CLUE_VERSION))
        s.merge_props("Nav", "sub_id", sid, dict(
            name_status="CURATED", curated_by=BY, curated_at=now, curation_batch=BATCH,
            superseded_name_provisional=row["name"],
            superseded_role_candidate=row["role"]))

    # 2) the 9 children of sub-17
    have = {r["sub_id"] for r in s.q("MATCH (n:Nav) RETURN n.sub_id AS sub_id")}
    for pid, (name, _) in sorted(PARTS.items()):
        role = CHILD_ROLE.get(pid, "SLICE")
        if pid not in have:
            s.create("Nav", dict(sub_id=pid, name=name, role=role, routable=True,
                                 parent=17, size=int(sizes[pid]),
                                 clue_version=CLUE_VERSION), "sub_id")
        else:
            s.conn.execute(
                "MATCH (n:Nav) WHERE n.sub_id = $pid SET n.name = $name, n.role = $role, "
                "n.parent = 17, n.size = $size, n.clue_version = $cv",
                parameters=dict(pid=pid, name=name, role=role,
                                size=int(sizes[pid]), cv=CLUE_VERSION))
        s.merge_props("Nav", "sub_id", pid, dict(
            name_status="CURATED", level="L2-child", parent_sub_id=17, from_candidate=17,
            curated_by=BY, curated_at=now, curation_batch=BATCH))
        s.conn.execute(
            "MATCH (p:Nav), (c:Nav) WHERE p.sub_id = 17 AND c.sub_id = $pid "
            "MERGE (p)-[:GuidesChild]->(c)", parameters=dict(pid=pid))

    # 3) membership: children gain Member edges; parent keeps its own (L1)
    for nid, pid in sorted(assign.items()):
        s.conn.execute(
            "MATCH (c:Nav), (n:Entity) WHERE c.sub_id = $pid AND n.nid = $nid "
            "MERGE (c)-[:Member]->(n)", parameters=dict(pid=pid, nid=nid))
        s.conn.execute("MATCH (n:Entity) WHERE n.nid = $nid SET n.curated_sub = $pid",
                       parameters=dict(nid=nid, pid=pid))
    # the ex-sub-18 node also joins the parent's own membership
    s.conn.execute(
        "MATCH (p:Nav), (n:Entity) WHERE p.sub_id = 17 AND n.sub = 18 "
        "MERGE (p)-[:Member]->(n)")

    # 4) merges: move membership, supersede the navigator, drop it from the master
    for src, (tgt, tname) in sorted(MERGED.items()):
        if src == 13:
            moved = [r["nid"] for r in s.q(
                "MATCH (a:Nav)-[:Member]->(n:Entity) WHERE a.sub_id = $src "
                "RETURN n.nid AS nid", dict(src=src))]
            for nid in sorted(moved):
                s.conn.execute(
                    "MATCH (b:Nav), (n:Entity) WHERE b.sub_id = $tgt AND n.nid = $nid "
                    "MERGE (b)-[:Member]->(n)", parameters=dict(tgt=tgt, nid=nid))
                s.conn.execute(
                    "MATCH (n:Entity) WHERE n.nid = $nid SET n.curated_sub = $tgt",
                    parameters=dict(nid=nid, tgt=tgt))
        # src's own Member edges are dropped either way (sub-18's were granted to
        # child 177 + parent 17 in phase 3)
        s.conn.execute(
            "MATCH (a:Nav)-[r:Member]->(:Entity) WHERE a.sub_id = $src DELETE r",
            parameters=dict(src=src))
        row = nav_row(s, src)
        s.create("ClueSnap", dict(
            snap_id=f"c2-{BATCH}-{src}", sub_id=src, taken_at=now, t_created=now,
            t_expired="", superseded_by=BATCH,
            body=jdump(snap_body(row, f"curation batch {BATCH}: merged into sub-{tgt}"))),
            "snap_id")
        s.conn.execute(
            "MATCH (a:Nav), (b:Nav) WHERE a.sub_id = $src AND b.sub_id = $tgt "
            "MERGE (a)-[:SupersededBy]->(b)", parameters=dict(src=src, tgt=tgt))
        s.conn.execute(
            "MATCH (a:Nav) WHERE a.sub_id = $src "
            "SET a.role = 'MERGED', a.clue_version = $cv, a.routable = false",
            parameters=dict(src=src, cv=CLUE_VERSION))
        s.merge_props("Nav", "sub_id", src, dict(
            name_status="CURATED", merged_into=tgt,
            superseded_name_provisional=row["name"],
            superseded_role_candidate=row["role"],
            supersession_reason=f"MERGE into sub-{tgt} ({tname})",
            curated_by=BY, curated_at=now, curation_batch=BATCH))
        s.conn.execute(
            "MATCH (m:Master)-[g:Guides]->(a:Nav) WHERE a.sub_id = $src DELETE g",
            parameters=dict(src=src))

    # 5) sizes on survivors + dissent annotations
    for r in s.q("MATCH (sn:Nav)-[:Member]->(n:Entity) WHERE sn.role <> 'MERGED' "
                 "RETURN sn.sub_id AS sid, count(n) AS c"):
        s.conn.execute("MATCH (sn:Nav) WHERE sn.sub_id = $sid SET sn.size = $c",
                       parameters=dict(sid=r["sid"], c=r["c"]))
    s.merge_props("Nav", "sub_id", 13, dict(curation_dissent=(
        "LAYER reading (90% Rule purity, LAYER_PURITY flag) considered and refuted: "
        "in-share 2.4% with 2 consumers is a consumer satellite, not a supplier layer. "
        "Retained here so the reading is queryable.")))
    rl = s.q("MATCH (n:Entity) WHERE n.name = 'real-login.ts' RETURN n.nid AS nid")
    for r in sorted(rl, key=lambda x: x["nid"]):
        s.merge_props("Entity", "nid", r["nid"], dict(curation_dissent=(
            "assignment_crossrepo_affinity=9 (BE auth journeys) argues for child 170 FE "
            "auth; kept in 173 FE fixtures & E2E harness on folder evidence "
            "(e2e-tests/_framework/). Owner question Q2.")))
    flagged = [r["nid"] for r in s.q(
        "MATCH (n:Entity) RETURN n.nid AS nid, n.props AS p")
        if json.loads(r["p"] or "{}").get("assignment_flagged") is True]
    for nid in sorted(flagged):
        s.merge_props("Entity", "nid", nid, dict(
            curation_verdict="CONFIRMED", curation_reviewed_at=now,
            curation_batch=BATCH))

    # 6) decision log (standalone rows; `batch` carries the master linkage — see header)
    for i, dec in enumerate(DECISIONS):
        s.create("CurationDecision", dict(
            did=f"c2-{BATCH}-{i:02d}-sub{dec['subsystem']}", batch=BATCH,
            decided_at=now, body=jdump(dict(dec, decided_by=BY))), "did")
    print(f"wrote {len(DECISIONS)} CurationDecision rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    main(ap.parse_args().dry)
