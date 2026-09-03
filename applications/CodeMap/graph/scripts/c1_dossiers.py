# CodeMap authoring layer — C1: per-subsystem dossiers from the CheckItOutV3 graph.
# Deterministic extraction, no judgement. The output is the curation agent's evidence base
# (see graph/prompts/GrothendieckPart2_CuratorV5.md and the design doc §C1/§C2).
#
# Usage:  PYTHONUTF8=1 python c1_dossiers.py [--out ../dossiers]
# Reads:  bolt://127.0.0.1:7611, namespace CheckItOutV3 (EntityDetail nodes, v4_subsystem)
# Writes: one JSON + one MD per subsystem + INDEX.md. Never writes to the graph.

import argparse, json, math, os, re, sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "authoring")))
from ladybug_store import Store

BOLT, AUTH, NS = "bolt://127.0.0.1:7611", ("neo4j", "password"), "CheckItOutV3"

# Behavioural edge set: everything except containment/detail/hyperedge/nav/violation plumbing.
DEPS = ["IMPORTS", "INJECTS", "EXTENDS", "CALLS", "USES", "PERFORMS", "ACCESSES",
        "IMPLEMENTS", "MODIFIES", "TRIGGERS", "VALIDATES", "AFFECTS", "TESTED_BY",
        "CONSTRAINS", "APPLIES_IN", "CONFIGURED_BY", "INITIATES"]

STOP = {"src", "main", "java", "app", "ts", "html", "scss", "com", "pl", "checkitout",
        "it", "io", "impl", "www", "the", "and", "for"}

LAYER_PURITY_MIN_N, LAYER_PURITY_THR, MEGA_SHARE, MICRO_N = 10, 0.85, 0.20, 3


def tokens(path, name):
    raw = re.split(r"[/\\._\-]", f"{path} {name}")
    out = []
    for t in raw:
        out += re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", t).lower().split()
    return [t for t in out if len(t) > 2 and t not in STOP and not t.isdigit()]


def trophic_heights(node_ids, edges):
    """MacKay levels on the internal digraph; per weakly-connected component, gauge min=0.
    Nodes with no internal behavioural edge get None (caller assigns layer median)."""
    idx = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)
    A = np.zeros((n, n))
    for a, b in edges:
        A[idx[a], idx[b]] += 1.0
    deg = A.sum(1) + A.sum(0)
    h = np.full(n, np.nan)
    # components on the undirected support
    seen, comps = set(), []
    und = (A + A.T) > 0
    for s in range(n):
        if s in seen or deg[s] == 0:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.nonzero(und[u])[0]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    for comp in comps:
        sub = np.ix_(comp, comp)
        Ac = A[sub]
        din, dout = Ac.sum(0), Ac.sum(1)
        L = np.diag(din + dout) - Ac - Ac.T
        v = np.linalg.lstsq(L, din - dout, rcond=None)[0]
        v -= v.min()
        for i, u in enumerate(comp):
            h[u] = v[i]
    return {node_ids[i]: (None if np.isnan(h[i]) else float(h[i])) for i in range(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "dossiers"))
    ap.add_argument("--curated", action="store_true")
    out_dir = os.path.abspath(ap.parse_args().out)
    os.makedirs(out_dir, exist_ok=True)

    curated = "--curated" in os.sys.argv if hasattr(os, "sys") else False
    import sys as _sys
    curated = "--curated" in _sys.argv
    s = Store(read_only=True)
    cur_map = {}
    if curated:
        cur_map = {r["nid"]: r["sub"] for r in s.q(
            "MATCH (sn:Nav)-[:Member]->(n:Entity) "
            "WHERE sn.role IS NULL OR NOT sn.role IN ['MERGED','GROUP'] "
            "RETURN n.nid AS nid, sn.sub_id AS sub")}
    nodes = s.q(
        "MATCH (n:Entity) WHERE n.sub IS NOT NULL RETURN n.nid AS id, n.name AS name, "
        "n.file_path AS path, n.entity_type AS et, n.sub AS v4, n.props AS props")
    nodes.sort(key=lambda r: r["id"])                   # determinism at the boundary
    for r in nodes:
        p = json.loads(r.pop("props") or "{}")
        r["loc"] = p.get("line_count", 0) or 0
        r["v3"] = p.get("v3_subsystem")
        r["bscore"] = float(p.get("boundary_score") or 0.0)
        r["isb"] = bool(p.get("is_boundary") or False)
    edges = [e for e in s.q(
        "MATCH (a:Entity)-[r:Dep]->(b:Entity) "
        "WHERE a.sub IS NOT NULL AND b.sub IS NOT NULL "
        "RETURN a.nid AS a, b.nid AS b, r.rel AS t") if e["t"] in DEPS]
    edges.sort(key=lambda e: (e["a"], e["b"], e["t"]))
    hyper = s.q(
        "MATCH (h:Hyperedge) RETURN h.key AS key, h.metapath AS mp, h.hub_name AS hub, "
        "h.arity AS arity, h.idf AS idf, h.prior AS prior, h.props AS hp")
    hyper.sort(key=lambda h: str(h["key"]))
    for h in hyper:  # member_ids ride in the props remainder (names have their own column)
        h["members"] = json.loads(h.pop("hp") or "{}").get("member_ids")
    emb = s.q("MATCH (n:Entity) WHERE n.sub IS NOT NULL "
              "RETURN n.nid AS id, n.sem_emb AS e")

    if curated:
        for r in nodes:
            r["v4"] = cur_map.get(r["id"], r["v4"])
    by_id = {r["id"]: r for r in nodes}
    total = len(nodes)
    subs = sorted({r["v4"] for r in nodes})
    members = {k: [r for r in nodes if r["v4"] == k] for k in subs}
    # COVERAGE ASSERT. In --curated mode the grouping is CONTAINS_MEMBER-from-a-leaf-navigator
    # with a v4_subsystem fallback (the coalesce is the .get default above), NOT a raw read of
    # the sparse curated_subsystem property — only 455 of 1415 nodes carry that, so grouping by
    # it alone would silently drop ~68% of the estate with no error. This assert makes that
    # class of mistake impossible to pass unnoticed.
    covered = sum(len(v) for v in members.values())
    assert covered == total, (
        f"COVERAGE FAIL: grouped {covered} of {total} nodes. In --curated mode every node must "
        f"land in exactly one group via CONTAINS_MEMBER-or-v4 fallback.")
    print(f"coverage OK: {covered} of {total} nodes grouped into {len(subs)} subsystems "
          f"{'(curated)' if curated else '(v4)'}")

    # cross/internal edges
    internal = defaultdict(list)          # sub -> [(a,b)]
    seams = defaultdict(Counter)          # sub -> Counter[(other, rel, dir)]
    ext_in = defaultdict(Counter)         # sub -> Counter[node_id] external in-degree
    int_in = defaultdict(Counter)         # sub -> Counter[node_id] internal in-degree
    ecount = defaultdict(lambda: [0, 0])  # sub -> [internal, external]
    for e in edges:
        sa, sb = by_id[e["a"]]["v4"], by_id[e["b"]]["v4"]
        if sa == sb:
            internal[sa].append((e["a"], e["b"]))
            int_in[sa][e["b"]] += 1
            ecount[sa][0] += 1
        else:
            seams[sa][(sb, e["t"], "out")] += 1
            seams[sb][(sa, e["t"], "in")] += 1
            ext_in[sb][e["b"]] += 1
            ecount[sa][1] += 1
            ecount[sb][1] += 1

    # tf-idf over subsystem token bags
    bags = {k: Counter(t for r in members[k] for t in tokens(r["path"], r["name"])) for k in subs}
    df = Counter(t for k in subs for t in bags[k])
    def top_terms(k, n=10):
        return [t for t, _ in sorted(
            ((t, c * math.log(len(subs) / df[t])) for t, c in bags[k].items()),
            key=lambda x: -x[1])[:n]]

    # medoids in semantic space
    evec = {r["id"]: np.array(r["e"], dtype=np.float32) for r in emb if r["e"]}
    def medoids(k, n=3):
        ids = [r["id"] for r in members[k] if r["id"] in evec]
        if not ids:
            return []
        M = np.stack([evec[i] / (np.linalg.norm(evec[i]) + 1e-9) for i in ids])
        c = M.mean(0)
        c /= np.linalg.norm(c) + 1e-9
        order = np.argsort(-(M @ c))
        return [by_id[ids[i]]["name"] for i in order[:n]]

    index_rows = []
    for k in subs:
        mem = members[k]
        n = len(mem)
        lp = Counter(r["et"] for r in mem)
        dom, domc = lp.most_common(1)[0]
        purity = domc / n
        flags = []
        if purity > LAYER_PURITY_THR and n >= LAYER_PURITY_MIN_N:
            flags.append("LAYER_PURITY")
        if n / total > MEGA_SHARE:
            flags.append("MEGA")
        if n < MICRO_N:
            flags.append("MICRO")

        heights = trophic_heights([r["id"] for r in mem], internal[k])
        hv = [v for v in heights.values() if v is not None]
        ids_in_k = {r["id"] for r in mem}
        entry = [{"name": by_id[i]["name"], "ext_in": c} for i, c in ext_in[k].most_common(5)]
        actor_roots = [r["name"] for r in mem
                       if r["et"] == "Actor" and int_in[k][r["id"]] == 0][:5]
        hs = []
        for h in hyper:
            inside = sum(1 for m in (h["members"] or []) if m in ids_in_k)
            if inside * 2 > (h["arity"] or len(h["members"] or [])):
                hs.append(h)
        hs.sort(key=lambda h: -(h["idf"] or 0))
        folders = Counter("/".join((r["path"] or "").replace("\\", "/").split("/")[:4]) for r in mem)

        d = {
            "subsystem": k, "size": n, "share": round(n / total, 4),
            "layer_profile": {t: c for t, c in lp.most_common()},
            "dominant_layer": dom, "purity": round(purity, 3), "flags": flags,
            "edges_internal": ecount[k][0], "edges_external": ecount[k][1],
            "external_ratio": round(ecount[k][1] / max(1, sum(ecount[k])), 3),
            "top_seams": [{"other": o, "rel": t, "dir": dr, "n": c}
                          for (o, t, dr), c in seams[k].most_common(3)],
            "entry_points": entry, "actor_roots": actor_roots,
            "trophic_span": [round(min(hv), 2), round(max(hv), 2)] if hv else None,
            "top_terms": top_terms(k), "medoids": medoids(k),
            "hyperedges_majority": len(hs),
            "top_hyperedges": [{"metapath": h["mp"], "hub": h["hub"], "arity": h["arity"],
                                "idf": round(h["idf"], 3) if h["idf"] else None} for h in hs[:3]],
            "boundary_mean": round(float(np.mean([r["bscore"] for r in mem])), 3),
            "boundary_nodes": sum(1 for r in mem if r["isb"]),
            "name_hint_folders": [f for f, _ in folders.most_common(2)],
            "v3_overlap": dict(Counter(r["v3"] for r in mem if r["v3"] is not None).most_common(2)),
            "stability_split_ari": None,
            "stability_note": "20-split partitions not persisted; recompute in C2 prep if a merge/split hinges on it",
            "disagreement_pairs": None,
            "disagreement_note": "332-pair shortlist not persisted as artifact; regenerate for C2 queue",
        }
        with open(os.path.join(out_dir, f"subsystem_{k}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        with open(os.path.join(out_dir, f"subsystem_{k}.md"), "w", encoding="utf-8") as f:
            f.write(f"# Subsystem {k} — dossier\n\n"
                    f"size {n} ({d['share']:.1%}) · dominant {dom} {purity:.0%} · "
                    f"ext-ratio {d['external_ratio']} · flags: {', '.join(flags) or 'none'}\n\n"
                    f"layers: {d['layer_profile']}\n\n"
                    f"terms: {', '.join(d['top_terms'])}\n\n"
                    f"medoids: {', '.join(d['medoids'])}\n\n"
                    f"entry: {', '.join(e['name'] + ' (' + str(e['ext_in']) + ')' for e in entry) or '—'}\n\n"
                    f"actor roots: {', '.join(actor_roots) or '—'}\n\n"
                    f"seams: {d['top_seams']}\n\n"
                    f"hyperedges (majority-in): {d['hyperedges_majority']}, top: {d['top_hyperedges']}\n\n"
                    f"folders: {d['name_hint_folders']}\nv3 overlap: {d['v3_overlap']}\n")
        index_rows.append((k, n, f"{d['share']:.1%}", dom, f"{purity:.0%}",
                           d["external_ratio"], ",".join(flags) or "—",
                           ", ".join(d["top_terms"][:3])))

    with open(os.path.join(out_dir, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write("# C1 dossiers — index (CheckItOutV3, v4 partition)\n\n"
                "| sub | n | share | dominant | purity | ext | flags | top terms |\n"
                "|---|---|---|---|---|---|---|---|\n")
        for r in sorted(index_rows, key=lambda x: -x[1]):
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    print(f"OK {len(subs)} dossiers -> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
