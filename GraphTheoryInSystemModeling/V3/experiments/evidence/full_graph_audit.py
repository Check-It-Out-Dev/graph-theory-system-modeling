"""G3 — re-measure every load-bearing number over the full graph, with repeated splits.

Owner's directive: compute over the whole graph, not a subsystem of it. Two
things needed checking.

FIRST, is the graph actually whole? It looked like only 41% of the frontend was
indexed -- 317 .ts nodes against 780 tracked. It is not a gap: 181 of those are
.spec.ts and 271 are the generated OpenAPI client under src/app/api. Against
HAND-WRITTEN source the coverage is 946/969 .java (97.6%) and 317/328 .ts
(96.6%). The graph is essentially the whole hand-written codebase. Worth stating
because it also bounds what the co-change oracle can see: commits touching specs
or generated clients contribute nothing, since those files have no node.

SECOND, and this is the real issue: every headline number in this paper rests on
ONE 50/50 split of the commit history. A single split gives a point estimate with
no error bar, and several conclusions in this arc turned on differences of 0.01
to 0.05. If those differences are inside the split-to-split variance they are not
findings.

So: 20 independent commit splits, full graph, every representation and every
partition, reported as mean +- sd. Anything whose margin over its competitor is
smaller than the spread of either is demoted on the spot.

Also swept: the MAX_COMMIT_FILES cutoff, which was fixed at 30 by assertion and
never checked. If the ranking moves with it, the cutoff was doing work it should
not have been.
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
N_SPLITS = 20
CUTOFFS = [10, 20, 30, 50, 100]
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
    if npos == 0 or nneg == 0:
        return np.nan
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
                "RETURN id(n) AS id, n.file_path AS fp, n.embedding AS emb, "
                "n.v3_subsystem AS sub, n.v3_module AS mod, n.subsystem_id AS old", ns=NS))
            edges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "RETURN id(a) AS s, id(b) AS t, type(r) AS k", ns=NS))
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
    S = np.array([r["emb"] for r in nodes], dtype=np.float64)
    S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
    sub = np.array([r["sub"] if r["sub"] is not None else -1 for r in nodes])
    mod = np.array([r["mod"] if r["mod"] is not None else -1 for r in nodes])
    old = np.array([r["old"] if r["old"] is not None else -1 for r in nodes])
    leaf = np.array([hash(repo_of[i] + "/" + rel_of[i].rsplit("/", 1)[0]) % 10 ** 9
                     for i in range(n)])
    d3 = np.array([hash(repo_of[i] + "/" +
                        "/".join(rel_of[i].rsplit("/", 1)[0].split("/")[:3])) % 10 ** 9
                   for i in range(n)])

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for e in edges:
        u, v = idx.get(e["s"]), idx.get(e["t"])
        if u is not None and v is not None and u != v:
            G.add_edge(u, v)
    A = nx.to_numpy_array(G, nodelist=range(n))
    deg = np.maximum(A.sum(axis=1, keepdims=True), 1.0)
    A2 = A + 0.5 * (A / deg) @ A

    pairs = [(a, b) for a, b in itertools.combinations(range(n), 2)
             if repo_of[a] == repo_of[b]]
    P = np.array(pairs)
    print(f"full graph: {n} nodes, {G.number_of_edges()} edges, "
          f"{len(pairs):,} within-repo pairs\n")

    # ---- read commit history once
    raw = {}
    by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}
    for rp in REPOS:
        out = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        blocks = []
        for b in out.split("\x01")[1:]:
            files = {l.strip() for l in b.splitlines() if l.strip()}
            ids = sorted({by_repo[rp][f] for f in files if f in by_repo[rp]})
            blocks.append((len(files), ids))
        raw[rp] = blocks

    def labels_for(cutoff, rng, half):
        lab = collections.Counter()
        for rp in REPOS:
            usable = [(nf, ids) for nf, ids in raw[rp] if 2 <= len(ids) <= cutoff]
            keep = rng.random(len(usable)) < 0.5
            for (nf, ids), k in zip(usable, keep):
                if k != half:
                    continue
                for a, b in itertools.combinations(ids, 2):
                    lab[(a, b)] += 1
        return np.array([1 if lab.get((a, b), 0) else 0 for a, b in pairs])

    s_sim = np.einsum("ij,ij->i", S[P[:, 0]], S[P[:, 1]])
    preds = {
        "content embedding (S)": s_sim,
        "graph 2-hop": A2[P[:, 0], P[:, 1]],
        "same directory (leaf)": (leaf[P[:, 0]] == leaf[P[:, 1]]).astype(float),
        "graph adjacency": A[P[:, 0], P[:, 1]],
    }
    parts = {
        "derived subsystems (18)": sub,
        "derived modules (126)": mod,
        "directory leaf (252)": leaf,
        "directory depth 3 (16)": d3,
        "incumbent subsystem_id": old,
    }

    print(f"REPRESENTATIONS — AUC over {N_SPLITS} independent commit splits\n")
    rng = np.random.default_rng(SEED)
    acc = collections.defaultdict(list)
    for t in range(N_SPLITS):
        y = labels_for(30, np.random.default_rng(SEED + t), False)
        for name, sc in preds.items():
            acc[name].append(auc(sc, y))
    print(f"{'predictor':<28}{'mean AUC':>10}{'sd':>8}{'min':>8}{'max':>8}")
    for name in preds:
        v = np.array(acc[name])
        print(f"{name:<28}{v.mean():>10.4f}{v.std():>8.4f}{v.min():>8.4f}{v.max():>8.4f}")

    print(f"\n\nPARTITIONS — coherence and modularity over {N_SPLITS} splits\n")
    coh_acc, mod_acc = collections.defaultdict(list), collections.defaultdict(list)
    for t in range(N_SPLITS):
        y = labels_for(30, np.random.default_rng(SEED + t), False)
        W = np.zeros((n, n))
        W[P[:, 0], P[:, 1]] = y
        W[P[:, 1], P[:, 0]] = y
        dg, m2 = W.sum(axis=1), W.sum()
        for name, lab in parts.items():
            same = lab[P[:, 0]] == lab[P[:, 1]]
            if same.sum() and (~same).sum():
                coh_acc[name].append(y[same].mean() / max(y[~same].mean(), 1e-12))
            q = 0.0
            for c in set(lab):
                m_ = lab == c
                q += W[np.ix_(m_, m_)].sum() / m2 - (dg[m_].sum() / m2) ** 2
            mod_acc[name].append(q)
    print(f"{'partition':<28}{'coherence':>11}{'sd':>7}{'modularity':>13}{'sd':>8}")
    for name in parts:
        c, q = np.array(coh_acc[name]), np.array(mod_acc[name])
        print(f"{name:<28}{c.mean():>10.2f}x{c.std():>7.2f}{q.mean():>13.4f}{q.std():>8.4f}")

    print("\n\nIS THE HEADLINE MARGIN BIGGER THAN THE NOISE?\n")
    a, b = np.array(mod_acc["derived subsystems (18)"]), np.array(mod_acc["directory leaf (252)"])
    d = a - b
    print(f"  derived subsystems vs directory leaf, modularity, paired over splits")
    print(f"    mean difference {d.mean():+.4f}, sd {d.std():.4f}, "
          f"splits where derived wins: {(d > 0).sum()}/{len(d)}")
    a3 = np.array(mod_acc["directory depth 3 (16)"])
    d3_ = a - a3
    print(f"  derived subsystems vs directory depth 3 (matched granularity)")
    print(f"    mean difference {d3_.mean():+.4f}, sd {d3_.std():.4f}, "
          f"splits where derived wins: {(d3_ > 0).sum()}/{len(d3_)}")
    am, aS = np.array(mod_acc["derived modules (126)"]), a
    dm = aS - am
    print(f"  subsystems vs modules (the F33 claim about natural scale)")
    print(f"    mean difference {dm.mean():+.4f}, sd {dm.std():.4f}, "
          f"splits where subsystems win: {(dm > 0).sum()}/{len(dm)}")

    print("\n\nCOMMIT-SIZE CUTOFF — was 30 doing work it should not have been?\n")
    print(f"{'cutoff':>8}{'positives':>11}{'content AUC':>13}{'derived Q':>12}{'dir leaf Q':>12}")
    for cut in CUTOFFS:
        y = labels_for(cut, np.random.default_rng(SEED), False)
        W = np.zeros((n, n))
        W[P[:, 0], P[:, 1]] = y
        W[P[:, 1], P[:, 0]] = y
        dg, m2 = W.sum(axis=1), W.sum()

        def q_of(lab):
            q = 0.0
            for c in set(lab):
                m_ = lab == c
                q += W[np.ix_(m_, m_)].sum() / m2 - (dg[m_].sum() / m2) ** 2
            return q

        print(f"{cut:>8}{int(y.sum()):>11,}{auc(s_sim, y):>13.4f}"
              f"{q_of(sub):>12.4f}{q_of(leaf):>12.4f}")
    print("\n  if the ranking holds across cutoffs, 30 was not load-bearing")


if __name__ == "__main__":
    main()
