# CodeMap diagnostic layer — measurable state IN the graph (SCHEMA.md section 6b).
# Computes the v1 functional catalogue, compares against the LAST stored observations,
# prints a drift report, writes the new observations (append-only, provenance-stamped).
# Since 2026-09-02 the authoring source is the Ladybug store (MIT) — same functionals,
# same drift law; observations land in the Diag table.
#
# Usage: PYTHONUTF8=1 python diag_state.py --run-id 2026-09-02-lb1 [--dry]

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "authoring")))
from c3_organisation import mackay_heights  # the one shared math implementation
from ladybug_store import Store

DRIFT_REL = 0.02  # relative drift below this is noise, above it gets flagged for a verdict


def compute(s):
    obs = []  # (functional, scope, value) — value float or json string
    nodes = s.q("MATCH (n:Entity) RETURN n.nid AS nid, n.name AS name, n.sub AS sub")
    edges = s.q("MATCH (a:Entity)-[r:Dep]->(b:Entity) "
                "RETURN a.nid AS aid, a.sub AS asub, b.nid AS bid, b.sub AS bsub")
    nodes.sort(key=lambda n: n["nid"])                      # determinism at the boundary
    edges.sort(key=lambda e: (e["aid"], e["bid"]))
    obs.append(("node_count", "global", float(len(nodes))))
    obs.append(("edge_count", "global", float(len(edges))))

    hc = s.q("MATCH (h:Hyperedge) RETURN h.source AS src, count(*) AS n")
    obs.append(("hyperedge_census", "global",
                json.dumps({r["src"]: r["n"] for r in sorted(hc, key=lambda x: str(x["src"]))})))

    lp = s.one("MATCH (n:Entity) RETURN "
               "sum(CASE WHEN n.sem_emb IS NOT NULL THEN 1 ELSE 0 END) AS s, "
               "sum(CASE WHEN n.beh_emb IS NOT NULL THEN 1 ELSE 0 END) AS b, "
               "sum(CASE WHEN n.str_emb IS NOT NULL THEN 1 ELSE 0 END) AS t")
    obs.append(("lens_presence", "global",
                json.dumps({"S": int(lp["s"]), "B": int(lp["b"]), "T": int(lp["t"])})))

    l2 = s.q("MATCH (sn:Nav) RETURN sn.clue_version AS v, count(*) AS n")
    obs.append(("l2_census", "global",
                json.dumps({r["v"]: r["n"] for r in sorted(l2, key=lambda x: str(x["v"]))})))
    snap = s.one("MATCH (c:ClueSnap) RETURN count(*) AS n")["n"]
    obs.append(("snapshot_count", "global", float(snap)))

    # navigation-tree reach, computed over the three rel pulls (Guides / GuidesChild /
    # Member are separate tables in the store — the mixed-type var-length match of the
    # Neo4j era becomes plain python set algebra, same numbers)
    roots = {r["s"] for r in s.q("MATCH (:Master)-[:Guides]->(n:Nav) RETURN n.sub_id AS s")}
    kids = defaultdict(set)
    for r in s.q("MATCH (a:Nav)-[:GuidesChild]->(b:Nav) RETURN a.sub_id AS p, b.sub_id AS c"):
        kids[r["p"]].add(r["c"])
    members = defaultdict(set)
    for r in s.q("MATCH (sn:Nav)-[:Member]->(e:Entity) RETURN sn.sub_id AS s, e.nid AS n"):
        members[r["s"]].add(r["n"])
    lvl2 = roots | {c for p in roots for c in kids[p]}
    reach = set().union(*(members[x] for x in lvl2)) if lvl2 else set()
    obs.append(("reachability", "global", float(len(set()
        .union(*(members[x] for x in roots)) if roots else set()))))
    # DEFECTIVE-GATE REPAIR (2026-09-02, ledger L5): `reachability` measures EXACTLY the
    # master's direct fan-out; after Q1=B made the tree 3-level it reports the two direct
    # roots' members (578) — stable by design. `reachability_le3` is the structural law.
    obs.append(("reachability_le3", "global", float(len(reach))))
    cur = s.q("MATCH (sn:Nav)-[:Member]->(n:Entity) "
              "WHERE sn.role IS NULL OR NOT sn.role IN ['MERGED','GROUP'] "
              "RETURN sn.sub_id AS sub, count(n) AS c")
    for r in sorted(cur, key=lambda x: x["sub"]):
        obs.append(("curated_subsystem_size", f"sub:{r['sub']}", float(r["c"])))

    # Vertex identity is the node id; the file name is a label — duplicate basenames are
    # DISTINCT vertices (2 colliding pairs live in sub-3). Same law as c3 / q_gold_all.
    by_sub, int_edges = defaultdict(list), defaultdict(list)
    ext = defaultdict(lambda: [0, 0])
    name_of = {n["nid"]: n["name"] for n in nodes}
    for n in nodes:
        if n["sub"] is not None:
            by_sub[n["sub"]].append(n["nid"])
    for e in edges:
        if e["asub"] == e["bsub"]:
            int_edges[e["asub"]].append((e["aid"], e["bid"]))
            ext[e["asub"]][0] += 1
        else:
            ext[e["asub"]][1] += 1
            ext[e["bsub"]][1] += 1
    for sub, ids in sorted(by_sub.items(), key=lambda kv: str(kv[0])):
        obs.append(("subsystem_size", f"sub:{sub}", float(len(ids))))
        i, x = ext[sub]
        obs.append(("external_ratio", f"sub:{sub}", round(x / max(1, i + x), 3)))
        hs = mackay_heights(ids, int_edges[sub])
        svc = sorted(v for nid, v in hs.items() if "Service" in name_of[nid] and v is not None)
        if svc:
            obs.append(("trophic_service_median", f"sub:{sub}", round(svc[len(svc) // 2], 2)))
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    s = Store()
    obs = compute(s)
    latest = {}
    for r in s.q("MATCH (d:Diag) RETURN d.functional AS f, d.scope AS sc, "
                 "d.value_num AS vn, d.value_json AS vj, d.computed_at AS at"):
        k = (r["f"], r["sc"])
        if k not in latest or str(r["at"]) > str(latest[k][0]):
            latest[k] = (r["at"], r["vn"] if r["vn"] is not None else r["vj"])
    prev = {k: v for k, (_, v) in latest.items()}
    drifts, new = [], 0
    for f, sc, v in obs:
        p = prev.get((f, sc))
        if p is None:
            new += 1
        elif isinstance(v, float) and isinstance(p, (int, float)):
            if abs(v - p) > DRIFT_REL * max(1.0, abs(p)):
                drifts.append(f"{f}[{sc}]: {p} -> {v}")
        elif str(p) != str(v):
            drifts.append(f"{f}[{sc}]: {p} -> {v}")
    print(f"functionals: {len(obs)} computed, {new} first-observation, {len(drifts)} drifted")
    for d in drifts:
        print("  DRIFT", d)
    if args.dry:
        return
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for f, sc, v in obs:
        s.create("Diag", dict(oid=f"{f}|{sc}|{args.run_id}", functional=f, scope=sc,
                              value_num=(v if isinstance(v, float) else None),
                              value_json=(v if isinstance(v, str) else None),
                              run_id=args.run_id, computed_at=now), pk="oid")
    chk = s.one("MATCH (d:Diag) WHERE d.run_id = $r RETURN count(*) AS c",
                dict(r=args.run_id))["c"]
    print(f"VERIFIED BY READ: {chk} observations written for run {args.run_id}")
    assert chk == len(obs)


if __name__ == "__main__":
    main()
