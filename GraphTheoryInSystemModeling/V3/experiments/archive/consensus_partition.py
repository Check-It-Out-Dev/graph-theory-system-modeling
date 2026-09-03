"""#51 — consensus of two strong, mechanistically different partitioners.

The position after #50: the champion (Louvain on content-kNN, 0.3555) and the
dense cohort-fiber partition (0.3462) are near-equal and built from DIFFERENT
signals — one from what files SAY (embedding similarity), one from what files DO
TOGETHER (train-half commit cohorts + meta-path fibers, an incidence structure).

Two facts argue the fusion should finally work:
  - F37: Strehl-Ghosh co-association reached 0.4029 modularity — the best number
    ever measured in this programme — with a far WEAKER second view (Leiden on
    typed edges, 0.18). It was never followed up.
  - F92's mechanism rule: combinations pay when the MECHANISMS differ over the
    same objects. Content vs incidence is exactly that; every failed combination
    was same-mechanism-over-subsets.

Construction per split (leakage discipline unchanged — fibers from the TRAIN
half only, evaluation on the TEST half only):
  S(i,j) = 0.5*[same cluster under champion] + 0.5*[same under cohort-fiber]
  Louvain on S, resolution tuned once on split 0 into the granularity band.

Registered bar unchanged: promotion needs >=18/20 paired wins over the champion
on held-out modularity. This is the first arm all day with a measured reason to
expect promotion rather than a hope.
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
KNN, N_SPLITS, SEED = 12, 20, 42
COHORT_RES = 2.0                  # tuned in #50
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
        for rp in REPOS:
            keep[rp] = rng.random(sum(1 for r, _ in commits if r == rp)) < 0.5
        marks, cnt = [], collections.defaultdict(int)
        for rp, ids in commits:
            marks.append(keep[rp][cnt[rp]])
            cnt[rp] += 1
        return np.array(marks)

    def y_from(mask, half):
        lab = collections.Counter()
        for (rp, ids), m in zip(commits, mask):
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

    def cohort_partition(mask):
        fibers = list(meta_fibers)
        for (rp, ids), m in zip(commits, mask):
            if m:
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
        return lab

    def consensus(labA, labB, res):
        CG = nx.Graph()
        CG.add_nodes_from(range(n))
        byA = collections.defaultdict(list)
        byB = collections.defaultdict(list)
        for i in range(n):
            byA[labA[i]].append(i)
            byB[labB[i]].append(i)
        seen = {}
        for grp in list(byA.values()):
            for a, b in itertools.combinations(sorted(grp), 2):
                if repo_of[a] == repo_of[b]:
                    seen[(a, b)] = 0.5
        for grp in list(byB.values()):
            for a, b in itertools.combinations(sorted(grp), 2):
                if repo_of[a] == repo_of[b]:
                    seen[(a, b)] = seen.get((a, b), 0.0) + 0.5
        for (a, b), w in seen.items():
            CG.add_edge(a, b, weight=w)
        comms = nx.community.louvain_communities(CG, weight="weight",
                                                 resolution=res, seed=SEED)
        lab = np.full(n, -1)
        for c, grp in enumerate(comms):
            for v in grp:
                lab[v] = c
        return lab

    # tune consensus resolution once on split 0
    m0 = split_masks(0)
    cf0 = cohort_partition(m0)
    # BISECTION between the percolation edges: the co-association graph jumps
    # 13 parts -> 315 parts between res 2.5 and 4.0, and a coarse ladder leaps
    # the band — which is the granularity confound this arc polices.
    lo, hi = 2.0, 4.0
    best = ((3.0, 0), 10 ** 9)
    for _ in range(10):
        mid = (lo + hi) / 2
        k = len(set(consensus(champ, cf0, mid)))
        print(f"  consensus res {mid:.3f}: {k} parts")
        gap = 0 if TARGET[0] <= k <= TARGET[1] else min(abs(k - TARGET[0]), abs(k - TARGET[1]))
        if gap < best[1]:
            best = ((mid, k), gap)
        if TARGET[0] <= k <= TARGET[1]:
            break
        if k < TARGET[0]:
            lo = mid
        else:
            hi = mid
    res_star, k0 = best[0]
    print(f"  chosen res = {res_star:.3f} ({k0} parts on split 0)\n")

    print("HELD-OUT BATTERY — consensus(champion, cohort-fiber), 20 splits\n")
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    kk = []
    lab_last = None
    for t in range(N_SPLITS):
        mask = split_masks(t)
        y = y_from(mask, False)
        cf = cohort_partition(mask)
        cons = consensus(champ, cf, res_star)
        kk.append(len(set(cons)))
        lab_last = cons
        for nm, l in (("champion", champ), ("cohort-fiber", cf), ("CONSENSUS", cons)):
            acc[nm]["mod"].append(modularity(l, y))
            acc[nm]["coh"].append(coherence(l, y))

    def tH(lab):
        tot = wsum = 0.0
        for c in set(lab):
            m = lab == c
            cnt = collections.Counter(et[m])
            k = m.sum()
            tot += -sum((v / k) * math.log2(v / k) for v in cnt.values() if v) * k
            wsum += k
        return tot / max(wsum, 1)

    base = np.array(acc["champion"]["mod"])
    print(f"{'arm':<16}{'parts':>7}{'coherence':>11}{'modularity':>12}{'type-H':>8}{'wins':>8}")
    for nm in ("champion", "cohort-fiber", "CONSENSUS"):
        m = np.array(acc[nm]["mod"])
        d = m - base
        wins = "—" if nm == "champion" else f"{int((d > 0).sum())}/{N_SPLITS}"
        kshow = len(set(champ)) if nm == "champion" else np.mean(kk) if nm == "CONSENSUS" else "~22"
        print(f"{nm:<16}{kshow if isinstance(kshow, str) else f'{kshow:.1f}':>7}"
              f"{np.nanmean(acc[nm]['coh']):>10.2f}x{m.mean():>12.4f}"
              f"{tH(champ if nm == 'champion' else lab_last):>8.3f}{wins:>8}")
    d = np.array(acc["CONSENSUS"]["mod"]) - base
    print(f"\n  CONSENSUS vs champion: Δ {d.mean():+.4f} ± {d.std():.4f}, "
          f"wins {int((d > 0).sum())}/{N_SPLITS}")
    print("  bar: >=18/20 promotes a NEW CHAMPION; anything less and A0 stands")


if __name__ == "__main__":
    main()
