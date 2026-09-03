"""#50 — commit-cohort fibers: the fiber algebra at the density it needed.

Every fiber-first failure had one root cause: 92 fibers covering a fifth of the
graph (F100, F101). This run multiplies the corpus by an order of magnitude with
V2's FEATURE_COHORT, finally given a source: each commit's file set is a fiber.

LEAKAGE DISCIPLINE, stated before the run, because the fibers now come from the
same history as the oracle: for every split, fibers are built from the TRAIN half
only and rebuilt per split; the held-out half never touches the skeleton. The 92
meta-path fibers are graph-derived and split-independent, so they join freely.

Pipeline per split:
  1. fibers = train-half commit cohorts (2..30 mapped nodes, weight 1/ln(2+size),
     large commits are weaker evidence per F50) + meta-path fibers (idf weight)
  2. fiber graph via inverted index (co-membership), Louvain -> fiber-clusters
  3. nodes take weighted-majority fiber-cluster membership; periphery attaches by
     label propagation over the content-kNN (the F100 rule)
  4. evaluate on the TEST half only: coherence + modularity, paired vs champion

Resolution for the fiber-graph Louvain is tuned ONCE on split 0 into the champion
granularity band and reused, so granularity is matched without per-split tuning.

Also probed (split 0 only): the alternating loop's weight drift on the dense
corpus. At 92 fibers it started at its fixed point (F101); with ~10x fibers the
purity re-weighting finally has room, and the drift number says whether the loop
is alive here.

The algebra point this run answers: after the magnetic Laplacian's double seal
(F99), the productive structure is not another operator on the sparse behavioural
digraph — any spectral operator shatters on a graph whose complement is 77%
isolated. It is the INCIDENCE structure of fibers: membership matrices, their
overlaps, and communities in the fiber dual. That is the algebra under test.
"""
import collections
import itertools
import math
import re
import subprocess

import numpy as np
import networkx as nx
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
KNN, N_SPLITS, SEED = 12, 20, 42
TARGET = (14, 26)


def main():
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
                "RETURN id(n) AS id, n.file_path AS fp, n.entity_type AS et, "
                "n.embedding AS emb", ns=NS))
            hyper = list(s.run(
                "MATCH (h:HyperedgeCandidate {namespace:$ns}) "
                "RETURN h.member_ids AS members, h.hub_id AS hub, h.idf_weight AS idf",
                ns=NS))
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
    X = np.array([r["emb"] for r in nodes], dtype=np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    # content kNN (champion base + propagation medium)
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
    champ = np.full(n, -1)
    for c, grp in enumerate(cm):
        for v in grp:
            champ[v] = c

    # commits
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
                commits.append((rp, ids))
    print(f"{len(commits)} usable commits; 92 meta-path fibers join split-independently")

    meta_fibers = []
    for h in hyper:
        mem = [idx[m] for m in h["members"] if m in idx]
        if h["hub"] in idx:
            mem.append(idx[h["hub"]])
        mem = sorted(set(mem))
        if len(mem) >= 2:
            meta_fibers.append((mem, max(h["idf"], 0.05)))

    pairs = [(a, b) for a, b in itertools.combinations(range(n), 2)
             if repo_of[a] == repo_of[b]]
    P = np.array(pairs)

    def split_masks(t):
        rng = np.random.default_rng(SEED + t)
        keep = {}
        pos = 0
        for rp in REPOS:
            k = rng.random(sum(1 for r, _ in commits if r == rp)) < 0.5
            keep[rp] = k
        marks = []
        cnt = collections.defaultdict(int)
        for rp, ids in commits:
            marks.append(keep[rp][cnt[rp]])
            cnt[rp] += 1
        return np.array(marks)

    def y_from(train_mask, half):
        lab = collections.Counter()
        for (rp, ids), m in zip(commits, train_mask):
            if m != half:
                continue
            for a, b in itertools.combinations(ids, 2):
                lab[(a, b)] += 1
        return np.array([1 if lab.get((a, b), 0) else 0 for a, b in pairs])

    def modularity(lab, y):
        W = np.zeros((n, n))
        W[P[:, 0], P[:, 1]] = y
        W[P[:, 1], P[:, 0]] = y
        k = W.sum(axis=1)
        m2 = W.sum()
        return float(sum(W[np.ix_(lab == c, lab == c)].sum() / m2 -
                         (k[lab == c].sum() / m2) ** 2 for c in set(lab)))

    def coherence(lab, y):
        same = lab[P[:, 0]] == lab[P[:, 1]]
        if not same.sum() or not (~same).sum() or y[~same].mean() == 0:
            return np.nan
        return float(y[same].mean() / y[~same].mean())

    def build_partition(train_mask, res):
        fibers = list(meta_fibers)
        for (rp, ids), m in zip(commits, train_mask):
            if m:
                fibers.append((ids, 1.0 / math.log(2 + len(ids))))
        # fiber graph via inverted index
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
                                                 resolution=res, seed=SEED)
        fcl = {f: c for c, grp in enumerate(comms) for f in grp}
        score = collections.defaultdict(lambda: collections.defaultdict(float))
        for fi, (mem, w) in enumerate(fibers):
            for m in mem:
                score[m][fcl[fi]] += w
        seed = {m: max(sc, key=sc.get) for m, sc in score.items()}
        lab = np.full(n, -1)
        for m, c in seed.items():
            lab[m] = c
        for _ in range(30):
            changed = 0
            for i in np.random.default_rng(SEED).permutation(n):
                if i in seed:
                    continue
                votes = collections.Counter()
                for j in knn_lists[i]:
                    if lab[j] >= 0:
                        votes[lab[j]] += W0[i, j]
                if votes:
                    best = max(votes, key=votes.get)
                    if best != lab[i]:
                        lab[i] = best
                        changed += 1
            if not changed:
                break
        if (lab < 0).any():
            big = collections.Counter(lab[lab >= 0]).most_common(1)[0][0]
            lab[lab < 0] = big
        return lab, fibers, fcl

    # resolution tuned once on split 0
    m0 = split_masks(0)
    res_star, k0 = 1.0, 0
    for res in (0.3, 0.5, 1.0, 2.0, 4.0):
        lab, _, _ = build_partition(m0, res)
        k = len(set(lab))
        print(f"  res {res}: induced {k} parts")
        res_star, k0 = res, k
        if TARGET[0] <= k <= TARGET[1]:
            break
    print(f"  chosen res = {res_star} ({k0} parts on split 0)\n")

    # alternation probe on split 0
    lab0, fibers0, fcl0 = build_partition(m0, res_star)
    drifts = []
    w_cur = [w for _, w in fibers0]
    for r in range(2):
        neww = []
        for (mem, w0), fi in zip(fibers0, range(len(fibers0))):
            cnt = collections.Counter(lab0[m] for m in mem)
            purity = cnt.most_common(1)[0][1] / len(mem)
            neww.append(max(0.05, w0 * purity))
        drifts.append(float(np.mean([abs(a - b) for a, b in zip(neww, w_cur)])))
        w_cur = neww
    print(f"alternation probe (split 0): weight drift per round {drifts}")
    print("  (F101 saw 0.044 -> 0.000 at 92 fibers; nonzero sustained drift = alive)\n")

    print("HELD-OUT BATTERY — fibers rebuilt from the TRAIN half of every split\n")
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    kk = []
    for t in range(N_SPLITS):
        mask = split_masks(t)
        y = y_from(mask, False)                    # test half only
        lab, _, _ = build_partition(mask, res_star)
        kk.append(len(set(lab)))
        for nm, l in (("champion", champ), ("cohort-fiber", lab)):
            acc[nm]["mod"].append(modularity(l, y))
            acc[nm]["coh"].append(coherence(l, y))
    base = np.array(acc["champion"]["mod"])
    test = np.array(acc["cohort-fiber"]["mod"])
    d = test - base

    def tH(lab):
        tot = wsum = 0.0
        for c in set(lab):
            m = lab == c
            cnt = collections.Counter(et[m])
            k = m.sum()
            tot += -sum((v / k) * math.log2(v / k) for v in cnt.values() if v) * k
            wsum += k
        return tot / max(wsum, 1)

    print(f"{'arm':<16}{'parts':>7}{'coherence':>11}{'modularity':>12}{'type-H':>8}")
    print(f"{'champion':<16}{len(set(champ)):>7}"
          f"{np.nanmean(acc['champion']['coh']):>10.2f}x{base.mean():>12.4f}"
          f"{tH(champ):>8.3f}")
    print(f"{'cohort-fiber':<16}{np.mean(kk):>7.1f}"
          f"{np.nanmean(acc['cohort-fiber']['coh']):>10.2f}x{test.mean():>12.4f}"
          f"{tH(lab0):>8.3f}")
    print(f"\n  paired: Δ modularity {d.mean():+.4f} ± {d.std():.4f}, "
          f"wins {int((d > 0).sum())}/{N_SPLITS}")
    print("  bar: >=18/20 to promote; 0/20 closes fiber-first at full density")


if __name__ == "__main__":
    main()
