# CodeMap Erdős E1 — internal organisation, extracted from the frozen recipe implementations
# (eval/scripts/q_gold_all.py: r_trophic_inversion / r_onboarding are the computation spec).
# Writes L3 navigation props per member: layer, local_height, entry_point, spine_membership.
# Conformance (prompt law): a live differential vs the frozen recipe — see the block in main().
#
# Usage: PYTHONUTF8=1 python c3_organisation.py [--dry]

import argparse, json, os, sys
from collections import Counter, defaultdict

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "authoring")))
from ladybug_store import Store

BOLT, AUTH, NS = "bolt://127.0.0.1:7611", ("neo4j", "password"), "CheckItOutV3"
RECIPE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval", "scripts"))
PROBES = (11, 4)  # the two subsystems diffed against the frozen recipe every run
REL = "IMPORTS|INJECTS|EXTENDS|CALLS|USES|PERFORMS|ACCESSES|IMPLEMENTS|MODIFIES|TRIGGERS|VALIDATES|AFFECTS|TESTED_BY|CONSTRAINS|APPLIES_IN|CONFIGURED_BY|INITIATES"


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

    GAUGE CHANGE (2026-09-02, task 63). Trophic height solves L h = d_in - d_out, and L is
    singular with nullity equal to the number of weakly-connected components, so h is
    determined only up to an INDEPENDENT additive constant PER COMPONENT. The previous single
    global shift imposed one origin on components that share no edges; worse, lstsq returns
    the minimum-norm solution, so each component's constant came from the pseudoinverse — an
    artifact, unstable under changes to unrelated components. c1_dossiers.py already gauged
    per component, which is how the disagreement surfaced (c3 >= c1 on every multi-component
    subsystem, never the reverse).

    Deliberately DUPLICATED (not imported) from eval/scripts/q_gold_all.py: the two stay
    independent implementations so the conformance differential below can still catch a
    divergence between them. Keep the two bodies identical by hand.
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


def mackay_heights(names, edges, with_components=False):
    idx = {n: i for i, n in enumerate(names)}
    A = np.zeros((len(names), len(names)))
    for a, b in edges:
        A[idx[a], idx[b]] += 1
    din, dout = A.sum(0), A.sum(1)
    deg = din + dout
    L = np.diag(deg) - A - A.T
    h = np.linalg.lstsq(L, din - dout, rcond=None)[0]
    comp = _component_labels(deg, A)
    h = _gauge_per_component(h, deg, comp)
    heights = {n: (float(h[i]) if deg[i] > 0 else None) for n, i in idx.items()}
    if with_components:
        return heights, {n: comp[i] for n, i in idx.items()}
    return heights


def main():
    dry = argparse.ArgumentParser().parse_args(
        ) if False else "--dry" in os.sys.argv
    s = Store()
    if True:
        nodes = s.q(
            "MATCH (n:Entity) WHERE n.sub IS NOT NULL "
            "RETURN n.nid AS nid, n.name AS name, n.entity_type AS et, n.sub AS sub")
        nodes.sort(key=lambda n: n["nid"])              # determinism at the boundary
        cur = {}
        if "--curated" in os.sys.argv:
            cur = {r["nid"]: r["sub"] for r in s.q(
                "MATCH (sn:Nav)-[:Member]->(n:Entity) "
                "WHERE sn.role IS NULL OR NOT sn.role IN ['MERGED','GROUP'] "
                "RETURN n.nid AS nid, sn.sub_id AS sub")}
            for r in nodes:
                r["sub"] = cur.get(r["nid"], r["sub"])
        edges = s.q(
            "MATCH (a:Entity)-[r:Dep]->(b:Entity) "
            "WHERE a.sub IS NOT NULL AND b.sub IS NOT NULL "
            "RETURN a.nid AS aid, a.name AS an, a.sub AS asub, "
            "b.nid AS bid, b.name AS bn, b.sub AS bsub")
        edges.sort(key=lambda e: (e["aid"], e["bid"]))
        # DEFECT REPAIR (2026-09-02, Erdos re-clue run). In curated mode the node->subsystem
        # remap was applied to NODES ONLY, so every edge kept its v4 endpoints while the node
        # set had already moved. Measured effect before this fix: int_edges[170..178] were
        # empty, so 0 of 406 frontend members received a local_height, and entry_point
        # degenerated to "every Actor in the child" — the flagged counts were exactly the Actor
        # counts 20/24/34/3/9/12/26/24/8. sub-4's heights were computed on the pre-merge
        # digraph, so the 49 absorbed ex-sub-13 files fell through to a layer median.
        # c1_dossiers.py already groups edges through the remapped node record (its by_id
        # lookup), so this aligns the two implementations rather than inventing a convention.
        # The non-curated path is untouched: `cur` is empty there and this loop is a no-op,
        # so the frozen-recipe conformance differential is unaffected.
        for e in edges:
            e["asub"] = cur.get(e["aid"], e["asub"])
            e["bsub"] = cur.get(e["bid"], e["bsub"])
        # source filter is prompt law (ledger L4): a script side-effect once flooded this label
        # with 1253 unweighted metapath-v2 candidates; spines must be built from v3 only.
        hyper = s.q(
            "MATCH (h:Hyperedge) WHERE h.metapath = 'A_P_R' AND h.source = 'metapath-v3' "
            "RETURN h.key AS key, h.hub_name AS hub, h.hub_nid AS hubid, "
            "h.hubsub AS hubsub, h.members AS members")
        hyper.sort(key=lambda h: str(h["key"]))
        for h in hyper:  # same remap, so a spine label names the curated owner of its hub
            h["members"] = json.loads(h["members"] or "null")
            h["hubsub"] = cur.get(h["hubid"], h["hubsub"])

        by_sub = defaultdict(list)
        for n in nodes:
            by_sub[n["sub"]].append(n)
        # int_edges carries NODE IDS, not names (2026-09-02, task 63 follow-up): two distinct
        # files can share a name (UnifiedStorageConfiguration.java, HashingUtilUnitTest.java,
        # both sub-3), and a name-keyed adjacency merges them into one vertex — it lost 2
        # vertices and distorted 66 of sub-3's 129 heights, the last c1/c3 disagreement.
        # Duplicated verbatim from eval/scripts/q_gold_all.py r_trophic_inversion. The
        # name-keyed ext_in/int_in counters below feed entry_point/actor-root selection and
        # are deliberately left alone — out of scope here, noted in the report.
        int_edges, ext_in, int_in = defaultdict(list), defaultdict(Counter), defaultdict(Counter)
        for e in edges:
            if e["asub"] == e["bsub"]:
                int_edges[e["asub"]].append((e["aid"], e["bid"]))
                int_in[e["asub"]][e["bn"]] += 1
            else:
                ext_in[e["bsub"]][e["bn"]] += 1

        spine_of = defaultdict(list)
        for h in hyper:
            for m in (h["members"] or []) + [h["hub"]]:
                spine_of[m].append(f"{h['hubsub']}:{h['hub']}")

        name_of = {n["nid"]: n["name"] for n in nodes}
        updates, conf, all_heights, all_comps = [], {}, {}, {}
        for sub, ms in by_sub.items():
            nids = [m["nid"] for m in ms]
            heights, all_comps[sub] = mackay_heights(
                nids, int_edges[sub], with_components=True)
            all_heights[sub] = heights
            layer_median = defaultdict(list)
            for m in ms:
                if heights[m["nid"]] is not None:
                    layer_median[m["et"]].append(heights[m["nid"]])
            layer_median = {k: sorted(v)[len(v) // 2] for k, v in layer_median.items()}
            # DETERMINISM LAW (2026-09-02, LB migration): most_common breaks ties by
            # Counter insertion order = backend edge-scan order. The tie-cutoff flipped
            # entry_point on deep files between the Neo4j-era pack and the Ladybug rerun
            # (measured: True on height-2.8 files). Name breaks ties, same rule as the
            # recipe's topn() — c3 and r_onboarding now select the SAME top-5.
            entries = {n for n, _ in sorted(ext_in[sub].items(),
                                            key=lambda kv: (-kv[1], kv[0]))[:5]}
            roots = {m["name"] for m in ms if m["et"] == "Actor"
                     and int_in[sub][m["name"]] == 0 and ext_in[sub][m["name"]] == 0}
            for m in ms:
                h = heights[m["nid"]]
                if h is None:
                    h = layer_median.get(m["et"])
                updates.append(dict(
                    nid=m["nid"], layer=m["et"],
                    height=round(h, 3) if h is not None else None,
                    entry=m["name"] in entries or m["name"] in roots,
                    spines=sorted(set(spine_of.get(m["name"], [])))))
            svc = sorted(v for nid, v in heights.items()
                         if "Service" in name_of[nid] and v is not None)
            if svc:
                conf[sub] = round(svc[len(svc) // 2], 2)

        # CONFORMANCE (prompt law; learnings ledger L2) — differential against the frozen
        # recipe implementation (eval/scripts/q_gold_all.py r_trophic_inversion) run on the
        # SAME pulled data: the recipe's own output rows (controller inversions) and its
        # service median must both reproduce from E1's heights, for two subsystems.
        #
        # This replaces a hardcoded `conf[11] == 0.94` anchor. That constant was correct on
        # the pre-delta 1374-node graph and went stale when delta batch 2026-09-02 re-indexed
        # 3 sub-11 files and moved its internal digraph: E1 and the frozen recipe BOTH moved
        # to 0.97, so E1 was right and the constant was wrong. A constant re-anchored by hand
        # after every delta degrades into a rubber stamp; a differential cannot.
        if "--curated" in os.sys.argv:
            print("conformance: SKIPPED in curated mode (recipe universe is v4; "
                  "run without --curated for the differential gate first)")
        else:
            _run_conformance = True
        sys.path.insert(0, RECIPE_DIR)
        import q_gold_all as recipe
        if "--curated" in os.sys.argv:
            recipe = None  # guarded above
        shim = type("G", (), {})()
        shim.nodes = [{"nid": n["nid"], "name": n["name"], "sub": n["sub"]} for n in nodes]
        shim.edges = edges
        # The comparison mirrors the recipe's COMPONENT-LOCAL semantics (task 63 parts b/c):
        # each controller against the median of its OWN weakly-connected component, and
        # controllers in Service-less components reported UNDEFINED rather than compared
        # against a foreign median. Rows carry both classes, so diffing the full row set
        # checks the gauge and the comparison in one assertion.
        for probe in (PROBES if "--curated" not in os.sys.argv else []):
            rows_r, ans_r, _ = recipe.r_trophic_inversion(shim, probe)
            hs, cs = all_heights[probe], all_comps[probe]
            svc_by_c = defaultdict(list)
            for nid, v in hs.items():
                if v is not None and "Service" in name_of[nid]:
                    svc_by_c[cs[nid]].append(v)
            med_by_c = {c: sorted(v)[len(v) // 2] for c, v in svc_by_c.items()}
            inv, und = [], []
            for nid, v in hs.items():
                n = name_of[nid]
                if v is None or "Controller" not in n:
                    continue
                c = cs[nid]
                if c not in med_by_c:
                    und.append(f"{n}|{v:.2f}|UNDEFINED_NO_SERVICE_IN_COMPONENT")
                elif v > med_by_c[c]:
                    inv.append(f"{n}|{v:.2f}|CONTROLLER_ABOVE_SERVICE_MEDIAN")
            mine = sorted(inv + und)
            theirs = sorted(f"{r[0]}|{r[1]}|{r[2]}" for r in rows_r if r[0] != "none")
            assert mine == theirs, \
                f"CONFORMANCE FAIL sub-{probe}\n  c3     {mine}\n  recipe {theirs}"
            meds = ", ".join(f"{m:.2f}" for _, m in sorted(med_by_c.items()))
            print(f"conformance OK sub-{probe}: {len(med_by_c)} component(s) with Services "
                  f"(medians {meds}), {len(inv)} inversion(s), {len(und)} undefined "
                  f"— zero drift vs frozen recipe")

        if dry:
            print(f"DRY: would write {len(updates)} nodes")
            return
        _v = "erdos-e1-curated" if "--curated" in os.sys.argv else "erdos-e1-v1"
        now = __import__("time").strftime("%Y-%m-%dT%H:%M:%S")
        for r in updates:
            # store dialect: entry_point rides as 'True'/'False' STRING; spines as an
            # in-query JSON literal (Store armor law — a bound list gets the notation cast)
            sp = json.dumps(r["spines"] or [], ensure_ascii=False)
            sp_lit = "'" + sp.replace("\\", "\\\\").replace("'", "\\'") + "'"
            params = dict(nid=r["nid"], layer=r["layer"] or "",
                          ep=str(bool(r["entry"])), ver=_v, at=now)
            if r["height"] is not None:  # a bound None is untypeable — NULL as literal
                h_clause, params["h"] = "n.local_height = $h", float(r["height"])
            else:
                h_clause = "n.local_height = NULL"
            s.conn.execute(
                f"MATCH (n:Entity) WHERE n.nid = $nid SET n.layer = $layer, "
                f"{h_clause}, n.entry_point = $ep, "
                f"n.spine_membership = {sp_lit}, n.org_version = $ver, n.org_at = $at",
                parameters=params)
        chk = s.one(
            "MATCH (n:Entity) WHERE n.org_version = $v RETURN count(*) AS c, "
            "sum(CASE WHEN n.entry_point = 'True' THEN 1 ELSE 0 END) AS entries, "
            "sum(CASE WHEN n.spine_membership <> '[]' THEN 1 ELSE 0 END) AS spined",
            dict(v=_v))
        print(f"VERIFIED BY READ: {chk['c']} nodes annotated, {chk['entries']} entry points, "
              f"{chk['spined']} spine members")
        assert chk["c"] == len(updates)


if __name__ == "__main__":
    main()
