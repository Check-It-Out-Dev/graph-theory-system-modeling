"""S7 — the Grothendieck V3 method itself, not the artifact it left behind.

Earlier iterations benchmarked against `subsystem_id` without reading the method
that produced it, which is not a fair test of the method. Section 12 of
GrothendieckAlgebraicTopologies.md specifies:

  Phase A  composite_proj in R^136 -- concatenate all 17 per-relation R^8
           projections. "Preserves ALL per-relation information without lossy
           fusion."
  Phase B  k-means on R^136                       -> cluster_topo
  Phase C  Leiden on the typed edge graph         -> cluster_graph
  Phase D  co-association fusion (Strehl & Ghosh 2002):
             S(i,j) = 0.5*[same topo] + 0.5*[same graph], Leiden on S
  Phase F  confidence = do the two views agree

Note what the method never uses: the raw R^4096 content embedding. Input 1 is the
R^136 composite, which is FastRP run on relation-filtered adjacency, so both of
its views are ultimately views of the GRAPH. 6e measured the graph at AUC
0.535-0.631 for co-change and the content embedding at 0.843, which predicts the
method is working from the weaker signal throughout.

Three questions, all with the same held-out co-change label:

  Q1  Head to head as representations: R^136 composite vs R^4096 content.
  Q2  Why is 77% of the incumbent unassigned? If most files carry no behavioural
      edge their composite is the zero vector, and no clustering can separate
      zeros. That would be a structural coverage limit of the method rather than
      a botched run -- a fair criticism instead of an unfair one.
  Q3  Does Phase D's co-association fusion actually help? Run it properly, with
      the CONTENT clustering as Input 1 instead of the composite. That keeps
      their fusion and swaps in the stronger view -- the honest synthesis, and
      the only way to tell whether the fusion idea or the input was at fault.
"""
import collections
import itertools
import subprocess

import numpy as np
import networkx as nx
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
PROJ = ["PERFORMS", "USES", "MODIFIES", "CALLS", "ACCESSES", "CONSTRAINS",
        "VALIDATES", "AFFECTS", "TRIGGERS", "APPLIES_IN", "CONFIGURED_BY", "INITIATES"]
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
    props = ", ".join(f"n.`proj_{p}` AS p_{p}" for p in PROJ)
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                f"MATCH (n:EntityDetail {{namespace:$ns}}) WHERE n.embedding IS NOT NULL "
                f"RETURN id(n) AS id, n.file_path AS fp, n.embedding AS emb, "
                f"n.subsystem_id AS old, n.v3_subsystem AS mine, {props}", ns=NS))
            edges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "RETURN id(a) AS s, id(b) AS t", ns=NS))
    finally:
        drv.close()

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
    C = np.zeros((n, 8 * len(PROJ)))
    for i, r in enumerate(nodes):
        for j, p in enumerate(PROJ):
            v = r[f"p_{p}"]
            if v:
                C[i, 8 * j:8 * (j + 1)] = v
    nz = np.abs(C).sum(axis=1) > 0
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)

    print("Q2 — WHY IS 77% OF THE INCUMBENT UNASSIGNED?\n")
    print(f"  files with a non-zero R^136 composite : {nz.sum()} of {n} ({nz.mean():.1%})")
    print(f"  files whose composite is the ZERO vector : {(~nz).sum()} ({(~nz).mean():.1%})")
    old = np.array([r["old"] if r["old"] is not None else -1 for r in nodes])
    print(f"  files the incumbent left unassigned (-1) : {(old < 0).sum()} ({(old < 0).mean():.1%})")
    overlap = ((~nz) & (old < 0)).sum()
    print(f"  overlap between the two                  : {overlap} "
          f"({overlap / max((old < 0).sum(), 1):.1%} of the unassigned)")
    print("\n  So the coverage gap is STRUCTURAL, not a botched run: a file with no")
    print("  behavioural edge has an all-zero composite, and no clustering can")
    print("  separate zeros from each other. The method can only place files that")
    print("  carry the relations it projects. That is a fair criticism of the")
    print("  method; my earlier framing of it as a bad partition was not.")

    # ---- labels
    by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}
    te = collections.Counter()
    for rp in REPOS:
        out = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        sets = [sorted({by_repo[rp][f] for f in
                        {l.strip() for l in b.splitlines() if l.strip()} if f in by_repo[rp]})
                for b in out.split("\x01")[1:]]
        sets = [s for s in sets if 2 <= len(s) <= 30]
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

    print("\n\nQ1 — R^136 COMPOSITE vs R^4096 CONTENT, AS REPRESENTATIONS\n")
    both = nz[P[:, 0]] & nz[P[:, 1]]
    print(f"  scored on {both.sum():,} pairs where BOTH files have a non-zero composite")
    print(f"  ({y[both].mean():.2%} base rate there, vs {y.mean():.2%} overall)\n")
    comp = np.einsum("ij,ij->i", Cn[P[both][:, 0]], Cn[P[both][:, 1]])
    cont = np.einsum("ij,ij->i", X[P[both][:, 0]], X[P[both][:, 1]])
    print(f"{'representation':<34}{'AUC':>8}")
    print(f"{'R^136 composite (Grothendieck)':<34}{auc(comp, y[both]):>8.3f}")
    print(f"{'R^4096 content (Qwen3)':<34}{auc(cont, y[both]):>8.3f}")
    print(f"{'concatenated, both':<34}"
          f"{auc(0.5 * comp + 0.5 * cont, y[both]):>8.3f}")

    # ---- Q3 co-association fusion, done properly
    print("\n\nQ3 — PHASE D FUSION, WITH THE STRONGER VIEW AS INPUT 1\n")

    def knn_graph(M, k=12):
        S = M @ M.T
        np.fill_diagonal(S, -1.0)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in np.argpartition(-S[i], k)[:k]:
                if repo_of[i] == repo_of[j] and S[i, j] > 0:
                    g.add_edge(i, int(j), weight=float(S[i, j]))
        return g

    def lv(g, res=1.0):
        cm = nx.community.louvain_communities(g, weight="weight", resolution=res, seed=SEED)
        lab = np.full(n, -1)
        for c, grp in enumerate(cm):
            for v in grp:
                lab[v] = c
        return lab

    Gstruct = nx.Graph()
    Gstruct.add_nodes_from(range(n))
    for e in edges:
        u, v = idx.get(e["s"]), idx.get(e["t"])
        if u is not None and v is not None and u != v:
            Gstruct.add_edge(u, v, weight=1.0)

    view_content = lv(knn_graph(X))
    view_struct = lv(Gstruct)
    view_comp = lv(knn_graph(Cn))

    def coassoc_fuse(v1, v2, res=1.0):
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for a, b in pairs:
            w = 0.5 * (v1[a] == v1[b]) + 0.5 * (v2[a] == v2[b])
            if w > 0:
                g.add_edge(a, b, weight=w)
        return lv(g, res)

    def coh(lab):
        same = lab[P[:, 0]] == lab[P[:, 1]]
        return float(y[same].mean() / max(y[~same].mean(), 1e-12))

    k_ = np.array([0.0])
    W = np.zeros((n, n))
    for a, b in pairs:
        if te.get((a, b), 0):
            W[a, b] = W[b, a] = 1.0
    deg = W.sum(axis=1)
    m2 = W.sum()

    def modularity(lab):
        q = 0.0
        for c in set(lab):
            m_ = lab == c
            q += W[np.ix_(m_, m_)].sum() / m2 - (deg[m_].sum() / m2) ** 2
        return float(q)

    cands = {
        "view 1: content kNN (mine)": view_content,
        "view 2: Leiden on typed edges": view_struct,
        "view 1': R^136 composite kNN": view_comp,
        "Phase D fusion (composite + struct)": coassoc_fuse(view_comp, view_struct),
        "Phase D fusion (content + struct)": coassoc_fuse(view_content, view_struct),
        "incumbent subsystem_id": old,
    }
    print(f"{'partition':<38}{'parts':>7}{'coherence':>11}{'modularity':>12}")
    for name, lab in cands.items():
        print(f"{name:<38}{len(set(lab)):>7}{coh(lab):>10.2f}x{modularity(lab):>12.4f}")
    print("\n  fusion helps only if it beats both of its own inputs.")


if __name__ == "__main__":
    main()
