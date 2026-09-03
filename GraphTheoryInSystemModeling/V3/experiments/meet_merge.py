"""#52 — meet-then-merge: the granularity-stable fusion operator.

F103 stated the open problem precisely: fusing two strong partitions by
re-clustering their co-association graph fails because that graph's community
structure is discontinuous in resolution. This is the operator built to replace
it, working directly on the partition lattice:

  MEET      A ∧ B — the intersection cells of champion A and cohort-fiber B.
            Every cell is a set of files BOTH views keep together, so the meet
            preserves every agreement and localises every disagreement.
  MERGE     agglomerate cells by the largest modularity gain on the TRAIN-half
            co-change graph, stopping at exactly k = |A| parts. Granularity is
            matched by construction — there is no resolution parameter to sit on
            a knife-edge, k just counts down.

Every final cluster is a union of meet cells: the operator never splits an
agreement, and resolves disagreements by evidence.

Leak discipline unchanged: B and the merge criterion see the TRAIN half only;
all reported numbers are from the TEST half.

THE ABLATION THAT KEEPS IT HONEST: the merge criterion uses train co-change, so
the fusion must beat "train co-change clustering alone" (Louvain on the train
graph, band-tuned) or the meet contributed nothing and this is just fitting
train history. Three arms: champion control, train-only Louvain, meet-merge.
Promotion bar unchanged: >=18/20 paired wins on held-out modularity.
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
COHORT_RES = 2.0


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
    K_TARGET = len(set(champ))

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

    def pair_labels(mask, half):
        lab = collections.Counter()
        for (rp, ids), m in zip(commits, mask):
            if m != half:
                continue
            for a, b in itertools.combinations(ids, 2):
                lab[(a, b)] += 1
        return lab

    def y_vec(lab):
        return np.array([1 if lab.get((a, b), 0) else 0 for a, b in pairs])

    def modularity(labv, y):
        W = np.zeros((n, n))
        W[P[:, 0], P[:, 1]] = y
        W[P[:, 1], P[:, 0]] = y
        k = W.sum(axis=1)
        m2 = W.sum()
        return float(sum(W[np.ix_(labv == c, labv == c)].sum() / m2 -
                         (k[labv == c].sum() / m2) ** 2 for c in set(labv)))

    def coherence(labv, y):
        same = labv[P[:, 0]] == labv[P[:, 1]]
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
            for m_ in mem:
                node_f[m_].append(fi)
        FE = collections.Counter()
        for m_, fl in node_f.items():
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
            for m_ in mem:
                score[m_][fcl[fi]] += w
        seedlab = {m_: max(sc, key=sc.get) for m_, sc in score.items()}
        labv = np.full(n, -1)
        for m_, c in seedlab.items():
            labv[m_] = c
        for _ in range(30):
            changed = 0
            for i in np.random.default_rng(SEED).permutation(n):
                if i in seedlab:
                    continue
                votes = collections.Counter()
                for j in knn_lists[i]:
                    if labv[j] >= 0:
                        votes[labv[j]] += W0[i, j]
                if votes:
                    best = max(votes, key=votes.get)
                    if best != labv[i]:
                        labv[i] = best
                        changed += 1
            if not changed:
                break
        if (labv < 0).any():
            big = collections.Counter(labv[labv >= 0]).most_common(1)[0][0]
            labv[labv < 0] = big
        return labv

    def meet_merge(A, B, train_lab, k_target):
        """Meet cells, then CNM-style agglomeration on train co-change to k_target."""
        cell_of = {}
        cells = {}
        for i in range(n):
            key = (A[i], B[i])
            if key not in cells:
                cells[key] = len(cells)
            cell_of[i] = cells[key]
        nc = len(cells)
        # cluster-level train edge weights and degree shares
        e = collections.defaultdict(float)
        deg = collections.defaultdict(float)
        tot = 0.0
        for (a, b), w in train_lab.items():
            ca, cb = cell_of[a], cell_of[b]
            w = float(w)
            tot += w
            deg[ca] += w
            deg[cb] += w
            if ca != cb:
                e[(min(ca, cb), max(ca, cb))] += w
        if tot == 0:
            return None, nc
        parent = list(range(nc))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        alive = set(range(nc))
        while len(alive) > k_target:
            best, bpair = -np.inf, None
            for (ca, cb), w in e.items():
                fa, fb = find(ca), find(cb)
                if fa == fb:
                    continue
                dq = 2 * (w / tot - (deg[fa] / (2 * tot)) * (deg[fb] / (2 * tot)))
                if dq > best:
                    best, bpair = dq, (fa, fb)
            if bpair is None:
                # no connected pairs left: merge two smallest-degree clusters
                srt = sorted(alive, key=lambda c: deg[c])
                bpair = (srt[0], srt[1])
            fa, fb = bpair
            parent[fb] = fa
            deg[fa] += deg[fb]
            alive.discard(fb)
            newe = collections.defaultdict(float)
            for (ca, cb), w in e.items():
                ra, rb = find(ca), find(cb)
                if ra != rb:
                    newe[(min(ra, rb), max(ra, rb))] += w
            e = newe
        labv = np.empty(n, dtype=int)
        for i in range(n):
            labv[i] = find(cell_of[i])
        return labv, nc

    def meet_quotient(A, B, train_lab, k_target, alpha):
        """v2 of the operator: Louvain on the CELL-QUOTIENT graph instead of
        greedy CNM (greedy agglomeration is order-unstable, which is where v1's
        ±0.044 split variance came from). The quotient carries BOTH mechanisms:
        train co-change counts and content-kNN weight between cells, each
        max-normalised, combined as T + alpha*C — the mechanism rule (F92) says
        different mechanisms combine over the same objects, and the cells are
        the objects. Resolution bisected to land exactly k_target parts."""
        cell_of = {}
        cells = {}
        for i in range(n):
            key = (A[i], B[i])
            if key not in cells:
                cells[key] = len(cells)
            cell_of[i] = cells[key]
        nc = len(cells)
        T = collections.defaultdict(float)
        C = collections.defaultdict(float)
        for (a, b), w in train_lab.items():
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
        Q.add_nodes_from(range(nc))
        for key in set(T) | set(C):
            w = T.get(key, 0.0) / tmax + alpha * C.get(key, 0.0) / cmax
            if w > 0:
                Q.add_edge(*key, weight=w)
        lo, hi = 0.1, 12.0
        best, gap = None, 10 ** 9
        for _ in range(14):
            mid = (lo + hi) / 2
            cmx = nx.community.louvain_communities(Q, weight="weight",
                                                   resolution=mid, seed=SEED)
            k = len(cmx)
            lab_c = {}
            for c, grp in enumerate(cmx):
                for f in grp:
                    lab_c[f] = c
            if abs(k - k_target) < gap:
                gap = abs(k - k_target)
                best = dict(lab_c)
            if k == k_target:
                break
            if k < k_target:
                lo = mid
            else:
                hi = mid
        labv = np.empty(n, dtype=int)
        for i in range(n):
            labv[i] = best[cell_of[i]]
        return labv

    def train_only(train_lab, k_target):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for (a, b), w in train_lab.items():
            G.add_edge(a, b, weight=float(w))
        lo, hi = 0.2, 8.0
        best = None
        for _ in range(12):
            mid = (lo + hi) / 2
            cmx = nx.community.louvain_communities(G, weight="weight",
                                                   resolution=mid, seed=SEED)
            big = sum(1 for c in cmx if len(c) > 1)
            labv = np.full(n, -1)
            for c, grp in enumerate(cmx):
                for v in grp:
                    labv[v] = c
            best = labv
            if abs(big - k_target) <= 3:
                break
            if big < k_target:
                lo = mid
            else:
                hi = mid
        return best

    print(f"champion k = {K_TARGET}; meet-merge targets the same k exactly\n")
    print("HELD-OUT BATTERY — 20 splits, paired\n")
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    ncells = []
    lastMM = None
    for t in range(N_SPLITS):
        mask = split_masks(t)
        tr = pair_labels(mask, True)
        y = y_vec(pair_labels(mask, False))
        B = cohort_partition(mask)
        MM, nc = meet_merge(champ, B, tr, K_TARGET)
        ncells.append(nc)
        TO = train_only(tr, K_TARGET)
        MQ = meet_quotient(champ, B, tr, K_TARGET, alpha=1.0)
        lastMM = MQ
        for nm, l in (("champion", champ), ("train-only Louvain", TO),
                      ("MEET-MERGE v1 (greedy)", MM),
                      ("MEET-QUOTIENT v2", MQ)):
            if l is None:
                continue
            acc[nm]["mod"].append(modularity(l, y))
            acc[nm]["coh"].append(coherence(l, y))

    def tH(labv):
        tot = wsum = 0.0
        for c in set(labv):
            m_ = labv == c
            cnt = collections.Counter(et[m_])
            k = m_.sum()
            tot += -sum((v / k) * math.log2(v / k) for v in cnt.values() if v) * k
            wsum += k
        return tot / max(wsum, 1)

    base = np.array(acc["champion"]["mod"])
    print(f"  meet produced {np.mean(ncells):.0f} cells on average "
          f"(agreements preserved, disagreements localised)\n")
    print(f"{'arm':<24}{'parts':>7}{'coherence':>11}{'modularity':>12}{'type-H':>8}{'wins':>8}")
    for nm in ("champion", "train-only Louvain", "MEET-MERGE v1 (greedy)",
               "MEET-QUOTIENT v2"):
        m = np.array(acc[nm]["mod"])
        d = m - base
        wins = "—" if nm == "champion" else f"{int((d > 0).sum())}/{N_SPLITS}"
        lv = champ if nm == "champion" else lastMM
        print(f"{nm:<24}{K_TARGET:>7}{np.nanmean(acc[nm]['coh']):>10.2f}x"
              f"{m.mean():>12.4f}{tH(lv):>8.3f}{wins:>8}")
    for nm in ("MEET-MERGE v1 (greedy)", "MEET-QUOTIENT v2"):
        d = np.array(acc[nm]["mod"]) - base
        dt = np.array(acc[nm]["mod"]) - np.array(acc["train-only Louvain"]["mod"])
        print(f"\n  {nm} vs champion   : Δ {d.mean():+.4f} ± {d.std():.4f}, "
              f"wins {int((d > 0).sum())}/{N_SPLITS}")
        print(f"  {nm} vs train-only : Δ {dt.mean():+.4f} ± {dt.std():.4f}, "
              f"wins {int((dt > 0).sum())}/{N_SPLITS}")
    print("\n  promotion needs >=18/20 vs champion AND a clear win vs train-only")
    print("  (else the fusion is just fitting train history)")


if __name__ == "__main__":
    main()
