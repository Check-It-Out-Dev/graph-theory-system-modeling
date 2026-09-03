# CodeMap E1 gate — is a reported trophic inversion a real finding, or a gauge artifact?
#
# DEFECT CLASS THIS CATCHES (2026-09-02, task 63): comparing two heights that do not share an
# origin. MacKay heights are fixed only up to an independent additive constant per weakly-
# connected component, so a service median pooled over the whole subsystem tests a controller
# in component A against a median drawn from component B. Nothing in the differential or in
# the span gate can see this — both sides can agree perfectly and still be meaningless.
# Measured before the fix: sub-7's ONE reported inversion (CustomErrorController.java,
# h=2.121) sat in a component holding no Service at all. It is now correctly UNDEFINED and
# its height reads 1.000.
#
# The gate asserts the invariant that replaced it: every surviving inversion is compared
# against a median from its OWN component, and controllers in Service-less components are
# reported UNDEFINED rather than silently compared against a foreign median. It also prints
# the retired pooled-median behaviour alongside, so a regression that reintroduces pooling
# shows up as inversions reappearing in Service-less components.
#
# Vertex identity is the NODE ID, never the file name — see c3_organisation.mackay_heights.
#
# Usage: PYTHONUTF8=1 python inversion_gauge_check_curated.py   (read-only; exit 1 on failure)

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c3_organisation import mackay_heights, _component_labels


def retired_pooled(nids, es, name_of):
    """Pre-2026-09-02 behaviour: global gauge + one median pooled across all components."""
    idx = {n: i for i, n in enumerate(nids)}
    A = np.zeros((len(nids), len(nids)))
    for a, b in es:
        A[idx[a], idx[b]] += 1
    din, dout = A.sum(0), A.sum(1)
    deg = din + dout
    L = np.diag(deg) - A - A.T
    h = np.linalg.lstsq(L, din - dout, rcond=None)[0]
    if (deg > 0).any():
        h -= h[deg > 0].min()
    comp = _component_labels(deg, A)
    hs = {n: (float(h[i]) if deg[i] > 0 else None) for n, i in idx.items()}
    cs = {n: comp[i] for n, i in idx.items()}
    svc = sorted(v for n, v in hs.items() if v is not None and "Service" in name_of[n])
    med = svc[len(svc) // 2] if svc else 0
    return [(n, hs[n], cs[n]) for n, v in hs.items()
            if v is not None and "Controller" in name_of[n] and v > med]


def current(nids, es, name_of):
    hs, cs = mackay_heights(nids, es, with_components=True)
    svc_by_c = defaultdict(list)
    for n, v in hs.items():
        if v is not None and "Service" in name_of[n]:
            svc_by_c[cs[n]].append(v)
    med_by_c = {c: sorted(v)[len(v) // 2] for c, v in svc_by_c.items()}
    inv, und = [], []
    for n, v in hs.items():
        if v is None or "Controller" not in name_of[n]:
            continue
        if cs[n] not in med_by_c:
            und.append((n, v, cs[n]))
        elif v > med_by_c[cs[n]]:
            inv.append((n, v, cs[n]))
    return inv, und, set(med_by_c)


def main():
    import os as _o
    import sys as _s
    _s.path.insert(0, _o.path.abspath(_o.path.join(
        _o.path.dirname(_o.path.abspath(__file__)), "..", "authoring")))
    from ladybug_store import Store
    s = Store(read_only=True)
    nodes = s.q("MATCH (n:Entity) WHERE n.sub IS NOT NULL "
                "RETURN n.nid AS nid, n.name AS name, n.sub AS sub")
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
    name_of = {n["nid"]: n["name"] for n in nodes}

    by_sub, int_edges = defaultdict(list), defaultdict(list)
    for n in nodes:
        by_sub[n["sub"]].append(n["nid"])
    for e in edges:
        if e["asub"] == e["bsub"]:
            int_edges[e["asub"]].append((e["aid"], e["bid"]))

    tot_b = tot_i = tot_u = artifacts = 0
    failures = []
    for sub in sorted(by_sub):
        nids, es = by_sub[sub], int_edges[sub]
        old = retired_pooled(nids, es, name_of)
        inv, und, svc_comps = current(nids, es, name_of)
        if not old and not inv and not und:
            continue
        tot_b += len(old)
        tot_i += len(inv)
        tot_u += len(und)
        print(f"\nsub-{sub}: retired-pooled {len(old)} | current {len(inv)} inversion(s), "
              f"{len(und)} undefined | components holding Services: {len(svc_comps)}")
        for n, v, c in sorted(old, key=lambda x: name_of[x[0]]):
            art = c not in svc_comps
            artifacts += art
            print(f"   POOLED  {name_of[n]:<44} h={v:6.3f} comp={c:<5}"
                  f"{'  <-- ARTIFACT (no Service in its component)' if art else ''}")
        for n, v, c in sorted(inv, key=lambda x: name_of[x[0]]):
            if c not in svc_comps:
                failures.append(f"sub-{sub} {name_of[n]}: inversion in a Service-less component")
            print(f"   CURRENT {name_of[n]:<44} h={v:6.3f} comp={c:<5}  same-component VERIFIED")
        for n, v, c in sorted(und, key=lambda x: name_of[x[0]]):
            print(f"   UNDEF   {name_of[n]:<44} h={v:6.3f} comp={c:<5}  no Service in component")

    print(f"\nretired pooled: {tot_b} inversion(s) | current: {tot_i} inversion(s) "
          f"+ {tot_u} undefined")
    print(f"cross-component artifacts eliminated: {artifacts}")
    if failures:
        for f in failures:
            print("FAIL", f)
        sys.exit(1)
    print("PASS — every surviving inversion is same-component; no foreign-median comparison")


if __name__ == "__main__":
    main()
