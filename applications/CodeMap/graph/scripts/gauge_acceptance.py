# CodeMap E1 gate — c1 and c3 must agree on trophic_span for EVERY curated leaf.
#
# WHY THIS EXISTS AS A PERMANENT GATE (2026-09-02, task 63). The conformance differential in
# c3_organisation.py diffs c3 against the frozen recipe on TWO probe subsystems (11 and 4).
# That is structurally unable to catch a defect the probes happen not to contain, and it did
# miss one: c3 keyed its adjacency by FILE NAME, so the two files in the graph that share a
# name (UnifiedStorageConfiguration.java, HashingUtilUnitTest.java — both in sub-3) collapsed
# into single vertices, losing 2 vertices and distorting 66 of sub-3's 129 heights. Neither
# probe contains a name collision, so the differential stayed green on luck, not correctness.
# This gate compares ALL leaves against c1, the independent implementation, and is what
# surfaced it. Two-probe differential + all-leaf agreement are complementary: keep both.
#
# It also proves LIVENESS, which the differential cannot: it recomputes each leaf under the
# retired global gauge as well, so a no-op edit to the gauge shows up as "0 leaves moved".
#
# Usage: PYTHONUTF8=1 python gauge_acceptance.py     (read-only; exit 1 on disagreement)

import json, os, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c3_organisation import mackay_heights

DOSS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dossiers"))


def retired_global_gauge(nids, edges):
    """The pre-2026-09-02 c3 behaviour: one global shift. Baseline for the liveness check."""
    idx = {n: i for i, n in enumerate(nids)}
    A = np.zeros((len(nids), len(nids)))
    for a, b in edges:
        A[idx[a], idx[b]] += 1
    din, dout = A.sum(0), A.sum(1)
    deg = din + dout
    L = np.diag(deg) - A - A.T
    h = np.linalg.lstsq(L, din - dout, rcond=None)[0]
    if (deg > 0).any():
        h -= h[deg > 0].min()
    return {n: (float(h[i]) if deg[i] > 0 else None) for n, i in idx.items()}


def main():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.path.abspath(_o.path.join(
        _o.path.dirname(_o.path.abspath(__file__)), "..", "authoring")))
    from ladybug_store import Store
    s = Store(read_only=True)
    nodes = s.q("MATCH (n:Entity) WHERE n.sub IS NOT NULL "
                "RETURN n.nid AS nid, n.sub AS sub")
    cur = {r["nid"]: r["sub"] for r in s.q(
        "MATCH (sn:Nav)-[:Member]->(n:Entity) "
        "WHERE sn.role IS NULL OR NOT sn.role IN ['MERGED','GROUP'] "
        "RETURN n.nid AS nid, sn.sub_id AS sub")}
    edges = s.q("MATCH (a:Entity)-[r:Dep]->(b:Entity) WHERE a.sub IS NOT NULL "
                "AND b.sub IS NOT NULL RETURN a.nid AS aid, a.sub AS asub, "
                "b.nid AS bid, b.sub AS bsub")
    nodes.sort(key=lambda n: n["nid"])
    edges.sort(key=lambda e: (e["aid"], e["bid"]))

    for r in nodes:
        r["sub"] = cur.get(r["nid"], r["sub"])
    for e in edges:
        e["asub"] = cur.get(e["aid"], e["asub"])
        e["bsub"] = cur.get(e["bid"], e["bsub"])

    by_sub, int_edges = defaultdict(list), defaultdict(list)
    for n in nodes:
        by_sub[n["sub"]].append(n["nid"])
    for e in edges:
        if e["asub"] == e["bsub"]:
            int_edges[e["asub"]].append((e["aid"], e["bid"]))

    doss = {}
    for f in os.listdir(DOSS):
        if f.startswith("subsystem_") and f.endswith(".json"):
            d = json.load(open(os.path.join(DOSS, f), encoding="utf-8"))
            doss[d["subsystem"]] = d

    print(f"{'sub':>5} {'size':>5} {'c1 dossier':>16} {'c3':>16} {'c3 (retired gauge)':>20}  status")
    agree = disagree = moved = 0
    for sub in sorted(by_sub):
        if sub not in doss:          # orphaned/superseded dossier ids are not leaves
            continue
        nids, es = by_sub[sub], int_edges[sub]
        hn = [v for v in mackay_heights(nids, es).values() if v is not None]
        ho = [v for v in retired_global_gauge(nids, es).values() if v is not None]
        if not hn:
            continue
        span_new = [round(min(hn), 2), round(max(hn), 2)]
        span_old = [round(min(ho), 2), round(max(ho), 2)]
        span_c1 = doss[sub].get("trophic_span")
        ok = span_c1 == span_new
        agree += ok
        disagree += (not ok)
        moved += (span_old != span_new)
        print(f"{sub:>5} {len(nids):>5} {str(span_c1):>16} {str(span_new):>16} "
              f"{str(span_old):>20}  {'OK' if ok else '*** DISAGREE ***'}")

    print(f"\nleaves compared: {agree + disagree} | AGREE {agree} | DISAGREE {disagree}")
    print(f"leaves whose span differs from the retired global gauge: {moved}")
    if disagree:
        print("ACCEPTANCE FAIL — c1 and c3 disagree; investigate before writing clues")
        sys.exit(1)
    print("ACCEPTANCE PASS")


if __name__ == "__main__":
    main()
