"""G5 — is 8 too few? Sweep the projection dimension over the full graph.

Owner's hypothesis, and a good one. Every sub-topology in this programme is an
R^8 projection of an R^4096 embedding whose effective dimensionality the
Information Lensing document puts at 50-200. If 8 starves the representation then
a great deal follows at once:

  * the R^136 composite losing to raw content 0.603 vs 0.844 (6g.3) would be a
    capacity artifact rather than evidence that the graph view is weak;
  * only 8 of 34 signatures carrying a reliable map (F11) would be partly a
    consequence of fitting in a space too small to separate them;
  * the near-orthogonality of relation subspaces (F8) would be what you get when
    every relation is crammed into the same 8 directions.

So sweep d over 8, 16, 32, 64, 128, 256 and measure at each: how well the
per-relation projections and their composite predict held-out git co-change, and
where that lands relative to the raw R^4096 content embedding at 0.844.

Full graph, every node, no sampling -- the machine has the memory for it. The
sweep runs FastRP at each dimension with everything else held fixed (same seed,
same iteration weights, same propertyRatio) so the ONLY thing varying is d.

Two outcomes are informative. If AUC climbs with d and approaches 0.844, then 8
was the bottleneck, the graph view was never given a fair test, and much of this
paper's pessimism about the typed graph needs revisiting. If it plateaus early
and low, the ceiling is the FastRP-over-adjacency construction itself and no
amount of width fixes it.
"""
import collections
import itertools
import subprocess

import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
RELS = ["PERFORMS", "USES", "MODIFIES", "CALLS", "ACCESSES",
        "IMPORTS", "INJECTS", "EXTENDS", "IMPLEMENTS", "TESTED_BY"]
DIMS = [8, 16, 32, 64, 128, 256]
GRAPH = "dimsweep"
SEED = 42


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
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    rng = np.random.default_rng(SEED)
    drv = GraphDatabase.driver(URI, auth=AUTH)

    with drv.session() as s:
        nodes = list(s.run("MATCH (n:EntityDetail {namespace:$ns}) "
                           "WHERE n.embedding IS NOT NULL "
                           "RETURN id(n) AS id, n.file_path AS fp, n.embedding AS emb",
                           ns=NS))
    n = len(nodes)
    idx = {r["id"]: i for i, r in enumerate(nodes)}
    repo_of, rel_of = {}, {}
    for i, r in enumerate(nodes):
        rest = (r["fp"] or "").replace("\\", "/")
        rest = rest[len(ROOT):] if rest.startswith(ROOT) else rest
        p = rest.split("/", 1)
        repo_of[i], rel_of[i] = (p[0], p[1]) if len(p) == 2 else ("?", rest)
    X = np.array([r["emb"] for r in nodes], dtype=np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    # ---- held-out co-change labels
    by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}
    te = collections.Counter()
    for rp in REPOS:
        out = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        sets = [sorted({by_repo[rp][f] for f in
                        {l.strip() for l in b.splitlines() if l.strip()} if f in by_repo[rp]})
                for b in out.split("\x01")[1:]]
        sets = [s_ for s_ in sets if 2 <= len(s_) <= 30]
        keep = rng.random(len(sets)) < 0.5
        for ids, k in zip(sets, keep):
            if k:
                continue
            for a, b in itertools.combinations(ids, 2):
                te[(a, b)] += 1
    pairs = [(a, b) for a, b in itertools.combinations(range(n), 2)
             if repo_of[a] == repo_of[b]]
    P = np.array(pairs)
    y = np.array([1 if te.get((a, b), 0) else 0 for a, b in pairs])
    print(f"full graph: {n} nodes, {len(pairs):,} within-repo pairs, "
          f"{y.sum():,} positives ({y.mean():.2%})\n")

    cont = np.einsum("ij,ij->i", X[P[:, 0]], X[P[:, 1]])
    print(f"reference — raw R^4096 content embedding: AUC {auc(cont, y):.4f}\n")

    # ---- project once, reuse for every dimension
    with drv.session() as s:
        s.run(f"CALL gds.graph.drop('{GRAPH}', false)")
        s.run(f"""
        MATCH (source:EntityDetail {{namespace:$ns}})-[r]->(target:EntityDetail {{namespace:$ns}})
        WHERE type(r) IN $rels
        WITH gds.graph.project('{GRAPH}', source, target, {{
            sourceNodeProperties: source {{ .embedding }},
            targetNodeProperties: target {{ .embedding }},
            relationshipType: type(r)
        }}, {{ undirectedRelationshipTypes: ['*'] }}) AS g
        RETURN g.nodeCount AS nodes, g.relationshipCount AS rels
        """, ns=NS, rels=RELS)

    def fastrp(graph, d, prop):
        with drv.session() as s:
            s.run(f"CALL gds.fastRP.mutate('{graph}', $c)", c={
                "embeddingDimension": d, "featureProperties": ["embedding"],
                "propertyRatio": 0.5, "iterationWeights": [0.0, 1.0, 1.0],
                "randomSeed": SEED, "mutateProperty": prop})
            rows = list(s.run(
                f"CALL gds.graph.nodeProperty.stream('{graph}', '{prop}') "
                f"YIELD nodeId, propertyValue RETURN nodeId, propertyValue"))
        M = np.zeros((n, d))
        for r in rows:
            i = idx.get(r["nodeId"])
            if i is not None:
                M[i] = r["propertyValue"]
        return M

    print(f"{'d':>5}{'base AUC':>11}{'composite AUC':>16}{'composite dim':>15}"
          f"{'vs content':>12}")
    results = []
    try:
        for d in DIMS:
            base = fastrp(GRAPH, d, f"b{d}")
            parts = []
            for k in RELS:
                gname = f"{GRAPH}_{k}_{d}"
                with drv.session() as s:
                    s.run(f"CALL gds.graph.drop('{gname}', false)")
                    s.run(f"CALL gds.graph.filter('{gname}', '{GRAPH}', '*', "
                          f"'r:{k}')")
                parts.append(fastrp(gname, d, f"p{d}"))
                with drv.session() as s:
                    s.run(f"CALL gds.graph.drop('{gname}', false)")
            comp = np.concatenate(parts, axis=1)

            def nrm(M):
                return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)

            bs = np.einsum("ij,ij->i", nrm(base)[P[:, 0]], nrm(base)[P[:, 1]])
            cs = np.einsum("ij,ij->i", nrm(comp)[P[:, 0]], nrm(comp)[P[:, 1]])
            a_b, a_c = auc(bs, y), auc(cs, y)
            results.append((d, a_b, a_c))
            print(f"{d:>5}{a_b:>11.4f}{a_c:>16.4f}{comp.shape[1]:>15}"
                  f"{a_c - auc(cont, y):>+12.4f}")
    finally:
        with drv.session() as s:
            s.run(f"CALL gds.graph.drop('{GRAPH}', false)")
        drv.close()

    if len(results) > 1:
        best = max(results, key=lambda r: r[2])
        first = results[0]
        print(f"\n  d=8 composite {first[2]:.4f}  ->  best d={best[0]} at {best[2]:.4f}"
              f"  ({best[2] - first[2]:+.4f})")
        print(f"  raw content embedding sits at {auc(cont, y):.4f}")
        if best[2] >= auc(cont, y) - 0.01:
            print("  => width WAS the bottleneck; the graph view was never fairly tested")
        elif best[2] - first[2] > 0.03:
            print("  => width helps materially but does not close the gap; the ceiling")
            print("     is partly the construction, partly the dimension")
        else:
            print("  => width is NOT the bottleneck; the ceiling is the")
            print("     FastRP-over-adjacency construction itself")


if __name__ == "__main__":
    main()
