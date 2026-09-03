"""#53 — promote MEET-QUOTIENT v2: build the production partition, write it back.

The registered criterion promoted a new champion (F106): meet of the content
partition and the cohort-fiber partition on the partition lattice, disagreements
resolved by Louvain on a cell-quotient carrying BOTH mechanisms (train co-change
incidence + content-kNN metric), granularity fixed at k by construction.
CV verdict over 20 held-out splits: modularity 0.3738 vs champion 0.3555,
18/20 paired wins against BOTH the champion and the train-only ablation.

This script does the standard final step: model selected by cross-validation,
production artifact fitted on ALL data (every commit feeds the fiber skeleton
and the quotient weights; no evaluation is claimed on it — the CV numbers are
the evaluation and travel with the provenance).

Writes: n.v4_subsystem on every node, and the promotion record on V3Master.
Old v3_subsystem properties are left untouched — supersession, never erasure.
"""
import collections
import itertools
import math
import subprocess

import numpy as np
import networkx as nx
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
KNN, SEED, COHORT_RES, ALPHA = 12, 42, 2.0, 1.0


def main():
    drv = GraphDatabase.driver(URI, auth=AUTH)
    with drv.session() as s:
        nodes = list(s.run(
            "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
            "RETURN id(n) AS id, n.file_path AS fp, n.entity_type AS et, "
            "n.embedding AS emb", ns=NS))
        hyper = list(s.run(
            "MATCH (h:HyperedgeCandidate {namespace:$ns}) "
            "RETURN h.member_ids AS members, h.hub_id AS hub, h.idf_weight AS idf",
            ns=NS))

    n = len(nodes)
    idx = {r["id"]: i for i, r in enumerate(nodes)}
    et = np.array([r["et"] for r in nodes])
    repo_of, rel_of = {}, {}
    for i, r in enumerate(nodes):
        rest = (r["fp"] or "").replace("\\", "/")
        rest = rest[len(ROOT):] if rest.startswith(ROOT) else rest
        p = rest.split("/", 1)
        repo_of[i], rel_of[i] = (p[0], p[1]) if len(p) == 2 else ("?", rest)
    X = np.array([r["emb"] for r in nodes], dtype=np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    Sq = X @ X.T
    np.fill_diagonal(Sq, -np.inf)
    W0 = np.zeros((n, n))
    knn_lists = {}
    for i in range(n):
        row = Sq[i].copy()
        row[[j for j in range(n) if repo_of[j] != repo_of[i]]] = -np.inf
        top = [int(j) for j in np.argpartition(-row, KNN)[:KNN]
               if np.isfinite(row[j]) and row[j] > 0]
        knn_lists[i] = top
        for j in top:
            W0[i, j] = W0[j, i] = max(W0[i, j], row[j])
    g0 = nx.Graph()
    g0.add_nodes_from(range(n))
    for a, b in np.argwhere(np.triu(W0, 1) > 0):
        g0.add_edge(int(a), int(b), weight=float(W0[a, b]))
    cm = nx.community.louvain_communities(g0, weight="weight", resolution=1.18, seed=SEED)
    A = np.full(n, -1)
    for c, grp in enumerate(cm):
        for v in grp:
            A[v] = c
    K = len(set(A))

    by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}
    commits = []
    for rp in REPOS:
        out = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        for b in out.split("\x01")[1:]:
            ids = sorted({by_repo[rp][f] for f in
                          {l.strip() for l in b.splitlines() if l.strip()}
                          if f in by_repo[rp]})
            if 2 <= len(ids) <= 30:
                commits.append(ids)

    fibers = []
    for h in hyper:
        mem = [idx[m] for m in h["members"] if m in idx]
        if h["hub"] in idx:
            mem.append(idx[h["hub"]])
        mem = sorted(set(mem))
        if len(mem) >= 2:
            fibers.append((mem, max(h["idf"], 0.05)))
    for ids in commits:
        fibers.append((ids, 1.0 / math.log(2 + len(ids))))

    node_f = collections.defaultdict(list)
    for fi, (mem, w) in enumerate(fibers):
        for m in mem:
            node_f[m].append(fi)
    FE = collections.Counter()
    for m, fl in node_f.items():
        if len(fl) > 1:
            for a, b in itertools.combinations(fl, 2):
                FE[(a, b)] += 1
    FG = nx.Graph()
    FG.add_nodes_from(range(len(fibers)))
    for (a, b), sh in FE.items():
        FG.add_edge(a, b, weight=sh * (fibers[a][1] + fibers[b][1]) / 2.0)
    comms = nx.community.louvain_communities(FG, weight="weight",
                                             resolution=COHORT_RES, seed=SEED)
    fcl = {f: c for c, grp in enumerate(comms) for f in grp}
    score = collections.defaultdict(lambda: collections.defaultdict(float))
    for fi, (mem, w) in enumerate(fibers):
        for m in mem:
            score[m][fcl[fi]] += w
    seedlab = {m: max(sc, key=sc.get) for m, sc in score.items()}
    B = np.full(n, -1)
    for m, c in seedlab.items():
        B[m] = c
    for _ in range(30):
        changed = 0
        for i in np.random.default_rng(SEED).permutation(n):
            if i in seedlab:
                continue
            votes = collections.Counter()
            for j in knn_lists[i]:
                if B[j] >= 0:
                    votes[B[j]] += W0[i, j]
            if votes:
                best = max(votes, key=votes.get)
                if best != B[i]:
                    B[i] = best
                    changed += 1
        if not changed:
            break
    if (B < 0).any():
        B[B < 0] = collections.Counter(B[B >= 0]).most_common(1)[0][0]

    # meet cells + two-mechanism quotient, all commits as evidence
    train = collections.Counter()
    for ids in commits:
        for a, b in itertools.combinations(ids, 2):
            train[(a, b)] += 1
    cells, cell_of = {}, {}
    for i in range(n):
        key = (A[i], B[i])
        if key not in cells:
            cells[key] = len(cells)
        cell_of[i] = cells[key]
    T = collections.defaultdict(float)
    C = collections.defaultdict(float)
    for (a, b), w in train.items():
        ca, cb = cell_of[a], cell_of[b]
        if ca != cb:
            T[(min(ca, cb), max(ca, cb))] += float(w)
    for a, b in np.argwhere(np.triu(W0, 1) > 0):
        ca, cb = cell_of[int(a)], cell_of[int(b)]
        if ca != cb:
            C[(min(ca, cb), max(ca, cb))] += float(W0[a, b])
    tmax = max(T.values(), default=1.0)
    cmax = max(C.values(), default=1.0)
    Q = nx.Graph()
    Q.add_nodes_from(range(len(cells)))
    for key in set(T) | set(C):
        w = T.get(key, 0.0) / tmax + ALPHA * C.get(key, 0.0) / cmax
        if w > 0:
            Q.add_edge(*key, weight=w)
    lo, hi = 0.1, 12.0
    best, gap = None, 10 ** 9
    for _ in range(14):
        mid = (lo + hi) / 2
        cmx = nx.community.louvain_communities(Q, weight="weight",
                                               resolution=mid, seed=SEED)
        k = len(cmx)
        lab_c = {f: c for c, grp in enumerate(cmx) for f in grp}
        if abs(k - K) < gap:
            gap, best = abs(k - K), dict(lab_c)
        if k == K:
            break
        lo, hi = (mid, hi) if k < K else (lo, mid)
    FINAL = np.array([best[cell_of[i]] for i in range(n)])
    ksz = collections.Counter(FINAL)
    print(f"production partition: {len(ksz)} subsystems over {n} files, "
          f"largest {max(ksz.values())}, meet cells {len(cells)}")

    def tH(labv):
        tot = wsum = 0.0
        for c in set(labv):
            m = labv == c
            cnt = collections.Counter(et[m])
            k = m.sum()
            tot += -sum((v / k) * math.log2(v / k) for v in cnt.values() if v) * k
            wsum += k
        return tot / max(wsum, 1)
    print(f"type-entropy {tH(FINAL):.3f} (ceiling 2.027); champion was 1.749")

    rows = [{"id": nodes[i]["id"], "s": int(FINAL[i])} for i in range(n)]
    with drv.session() as s:
        s.run("""
        UNWIND $rows AS r MATCH (m) WHERE id(m) = r.id SET m.v4_subsystem = r.s
        """, rows=rows)
        s.run("""
        MERGE (m:V3Master {namespace:$ns})
        SET m.v4_method = 'MEET-QUOTIENT v2: meet(content-Louvain, cohort-fiber) on the partition
lattice; cell-quotient Louvain with T/tmax + alpha*C/cmax edge weights; k matched to content champion',
            m.v4_parents = ['content-kNN Louvain res 1.18', 'cohort-fiber (commit cohorts + 92 metapath fibers, res 2.0)'],
            m.v4_alpha = $alpha, m.v4_k = $k, m.v4_meet_cells = $cells,
            m.v4_cv_modularity = 0.3738, m.v4_cv_vs_champion = '+0.0182 +- 0.0129, 18/20',
            m.v4_cv_vs_trainonly = '+0.0336 +- 0.0324, 18/20',
            m.v4_promoted_at = datetime(), m.v4_supersedes = 'v3_subsystem (kept)',
            m.v4_provenance = 'F102 density, F103 knife-edge, F105 greedy variance, F106 promotion; evaluation = 20 held-out commit splits, criterion pre-registered'
        RETURN m.v4_k
        """, ns=NS, alpha=ALPHA, k=len(ksz), cells=len(cells))
        chk = list(s.run(
            "MATCH (m:EntityDetail {namespace:$ns}) WHERE m.v4_subsystem IS NOT NULL "
            "RETURN count(*) AS c, count(DISTINCT m.v4_subsystem) AS k", ns=NS))
    drv.close()
    print(f"verified by read: {chk[0]['c']} nodes assigned across {chk[0]['k']} subsystems")


if __name__ == "__main__":
    main()
