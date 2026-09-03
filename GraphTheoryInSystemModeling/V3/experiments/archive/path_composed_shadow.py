"""V7b — do sub-topology traces recombine along the path?

Owner's proposal, and the correct form of what §6a/§6b was reaching for: as a walk
proceeds along a meta-path, each hop should be read through ITS OWN relation's
shadow, and the traces recombine along the way.

    u --k₁--> m --k₂--> v

§6a composed maps around loops in RELATION space — abstract, detached from the
graph, and it died with the rotation model class. This composes along an ACTUAL
PATH in the graph. It needs no rotation (the model-class ladder selected scalar
and diagonal, F15), it is anchored to real structure, and it asks the question
only where signal is already known to exist: the pairs behavioural meta-paths
connect at 11.9× lift.

The test isolates exactly one variable. Take every meta-path instance (u, m, v).
Score the endpoint pair three ways, all using the same walk:

    path-composed   sim(ρ_k₁(u), ρ_k₁(m)) · sim(ρ_k₂(m), ρ_k₂(v))
                    each hop evaluated in its own relation's shadow
    base-composed   sim(ρ₀(u), ρ₀(m)) · sim(ρ₀(m), ρ₀(v))
                    the same walk, every hop in the common shadow
    content         cosine of the raw embeddings, ignoring the walk

If the traces genuinely recombine, path-composed beats base-composed: reading a
hop through the relation that made it should carry information that the averaged
shadow does not. If they tie, the per-relation shadows add nothing over ρ₀ even
where the walk is real, and the sub-topologies are decoration everywhere.

Evaluated only among meta-path-connected pairs, where the co-change base rate is
30-49% rather than 3%, so the comparison is well-powered rather than drowning in
negatives. Bootstrapped over the pair set, since here the split is over pairs
rather than commits.
"""
import collections
import itertools
import subprocess

import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
N_BOOT, SEED = 500, 42

# (name, hop1 relation, mid type, hop2 relation, hop2 reversed?)
#
# The reversal flag matters and its absence was a real bug: a SYMMETRIC walk like
# P-USES-R-USES-P means two Processes pointing INTO the same Resource, so the
# second hop runs backwards. Written forward-forward it requires a Resource with
# outgoing USES edges, and USES is type-pure Process->Resource (F9), so it yields
# exactly zero pairs. Three of five walks silently produced nothing before this
# was fixed, costing two thirds of the available statistical power.
WALKS = [
    ("A-PERFORMS-P-USES-R", "PERFORMS", "Process", "USES", False),
    ("A-PERFORMS-P-MODIFIES-R", "PERFORMS", "Process", "MODIFIES", False),
    ("A-PERFORMS-P-ACCESSES-R", "PERFORMS", "Process", "ACCESSES", False),
    ("P-USES-R-USES-P", "USES", "Resource", "USES", True),
    ("P-MODIFIES-R-MODIFIES-P", "MODIFIES", "Resource", "MODIFIES", True),
    ("A-PERFORMS-P-PERFORMS-A", "PERFORMS", "Process", "PERFORMS", True),
    ("A-ACCESSES-R-ACCESSES-A", "ACCESSES", "Resource", "ACCESSES", True),
    ("P-USES-R-MODIFIES-P", "USES", "Resource", "MODIFIES", True),
]


def auc(scores, labels):
    order = np.argsort(scores, kind="mergesort")
    s, y = np.asarray(scores, float)[order], np.asarray(labels)[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    npos, nneg = float(y.sum()), float((1 - y).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    rels = sorted({w[1] for w in WALKS} | {w[3] for w in WALKS if w[3]})
    props = ", ".join(f"n.`proj_{r}` AS p_{r}" for r in rels)
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                f"MATCH (n:EntityDetail {{namespace:$ns}}) WHERE n.rho0 IS NOT NULL "
                f"RETURN id(n) AS id, n.file_path AS fp, n.entity_type AS et, "
                f"n.rho0 AS rho0, n.embedding AS emb, {props}", ns=NS))
            edges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "WHERE type(r) IN $rels RETURN id(a) AS s, id(b) AS t, type(r) AS k",
                ns=NS, rels=rels))
    finally:
        drv.close()

    n = len(nodes)
    idx = {r["id"]: i for i, r in enumerate(nodes)}
    et = np.array([r["et"] for r in nodes])
    repo_of, rel_of = {}, {}
    for i, r in enumerate(nodes):
        rest = (r["fp"] or "").replace("\\", "/")
        rest = rest[len(ROOT):] if rest.startswith(ROOT) else rest
        p = rest.split("/", 1)
        repo_of[i], rel_of[i] = (p[0], p[1]) if len(p) == 2 else ("?", rest)

    def unit(M):
        return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)

    R0 = unit(np.array([r["rho0"] for r in nodes], dtype=np.float64))
    EMB = unit(np.array([r["emb"] for r in nodes], dtype=np.float64))
    PROJ = {}
    for rel in rels:
        M = np.array([r[f"p_{rel}"] if r[f"p_{rel}"] else [0.0] * 8 for r in nodes],
                     dtype=np.float64)
        PROJ[rel] = (unit(M), np.abs(M).sum(axis=1) > 0)

    out_by_rel = collections.defaultdict(lambda: collections.defaultdict(list))
    in_by_rel = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in edges:
        u, v = idx.get(e["s"]), idx.get(e["t"])
        if u is not None and v is not None and u != v:
            out_by_rel[e["k"]][u].append(v)
            in_by_rel[e["k"]][v].append(u)

    # co-change labels (all commits; the split here is over pairs, not commits)
    by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}
    lab = collections.Counter()
    for rp in REPOS:
        txt = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        for b in txt.split("\x01")[1:]:
            ids = sorted({by_repo[rp][f] for f in
                          {l.strip() for l in b.splitlines() if l.strip()}
                          if f in by_repo[rp]})
            if 2 <= len(ids) <= 30:
                for a, c in itertools.combinations(ids, 2):
                    lab[(a, c)] += 1

    def cos(M, a, b):
        return float(np.dot(M[a], M[b]))

    print("V7b — DOES READING EACH HOP IN ITS OWN SHADOW BEAT THE COMMON SHADOW?\n")
    grand = {"path": [], "base": [], "content": [], "y": []}
    for name, k1, midtype, k2, rev2 in WALKS:
        triples = []
        for u, mids in out_by_rel.get(k1, {}).items():
            for m in mids:
                if midtype and et[m] != midtype:
                    continue
                nxt = (in_by_rel if rev2 else out_by_rel).get(k2, {}).get(m, ())
                for v in nxt:
                    if v == u or repo_of[u] != repo_of[v]:
                        continue
                    triples.append((u, m, v))
        # one row per endpoint pair, keeping the best-connecting intermediate
        best = {}
        for u, m, v in triples:
            key = (min(u, v), max(u, v))
            if key not in best:
                best[key] = (u, m, v)
        if len(best) < 30:
            print(f"  {name:<26} only {len(best)} pairs, skipped")
            continue

        P1, s1 = PROJ[k1]
        P2, s2 = PROJ[k2]
        rows = []
        for (a, b), (u, m, v) in best.items():
            if not (s1[u] and s1[m] and s2[m] and s2[v]):
                continue
            path = cos(P1, u, m) * cos(P2, m, v)
            base = cos(R0, u, m) * cos(R0, m, v)
            cont = cos(EMB, u, v)
            rows.append((path, base, cont, 1 if lab.get((a, b), 0) else 0))
        if len(rows) < 30 or len({r[3] for r in rows}) < 2:
            print(f"  {name:<26} {len(rows)} usable pairs, not enough signal")
            continue
        pa, ba, co, y = (np.array([r[i] for r in rows]) for i in range(4))
        grand["path"].append(pa); grand["base"].append(ba)
        grand["content"].append(co); grand["y"].append(y)
        print(f"  {name:<26} {len(rows):>5} pairs, {y.mean():>5.1%} co-change   "
              f"path {auc(pa, y):.3f}  base {auc(ba, y):.3f}  content {auc(co, y):.3f}")

    if not grand["y"]:
        print("\n  no walk produced enough usable pairs")
        return
    pa = np.concatenate(grand["path"]); ba = np.concatenate(grand["base"])
    co = np.concatenate(grand["content"]); y = np.concatenate(grand["y"])
    print(f"\n\nPOOLED — {len(y)} meta-path-connected pairs, "
          f"{y.mean():.1%} co-change base rate\n")
    print(f"{'scorer':<40}{'AUC':>8}")
    print(f"{'path-composed (each hop own shadow)':<40}{auc(pa, y):>8.4f}")
    print(f"{'base-composed (all hops rho0)':<40}{auc(ba, y):>8.4f}")
    print(f"{'content embedding (ignores the walk)':<40}{auc(co, y):>8.4f}")

    rng = np.random.default_rng(SEED)
    wins = 0
    diffs = []
    for _ in range(N_BOOT):
        s = rng.integers(0, len(y), len(y))
        if len(set(y[s])) < 2:
            continue
        d = auc(pa[s], y[s]) - auc(ba[s], y[s])
        diffs.append(d)
        wins += d > 0
    diffs = np.array(diffs)
    print(f"\n  path-composed minus base-composed, {len(diffs)} bootstrap resamples:")
    print(f"    mean {diffs.mean():+.4f}   sd {diffs.std():.4f}   "
          f"wins {wins}/{len(diffs)} ({wins/len(diffs):.0%})")
    print(f"    95% interval [{np.percentile(diffs,2.5):+.4f}, "
          f"{np.percentile(diffs,97.5):+.4f}]")
    verdict = ("traces DO recombine — the per-relation shadow beats the common one"
               if np.percentile(diffs, 2.5) > 0 else
               "no evidence the per-relation shadow beats rho0 along the walk")
    print(f"\n  => {verdict}")


if __name__ == "__main__":
    main()
