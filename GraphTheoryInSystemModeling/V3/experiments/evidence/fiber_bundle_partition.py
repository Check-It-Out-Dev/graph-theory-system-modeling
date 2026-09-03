"""#49 — fibers, trophic phase, and the alternating loop.

Three fixes, each aimed at a MEASURED failure rather than at an intuition.

FIX 1 — periphery attachment (A4's localised defect).
    A4 built the most vertical partition of any arm (type-H 1.856) and then lost
    everything at the attachment stage: 314 nodes sit on fibers, 77% are
    periphery, and top-3 mean cosine dragged them anywhere. Replace with LABEL
    PROPAGATION over the content-kNN seeded by fiber cores — the graph fleshing
    out a topological skeleton instead of a global similarity vote. This is the
    owner's architecture: not topology-then-graph, but each enhancing the other.

FIX 2 — the magnetic Laplacian, with a phase that is not fitted.
    Its measured failure (F16) was a phase fitted from an arbitrary real FastRP
    basis: nothing to encode, nothing recovered. The literature separates the two
    hierarchies cleanly — magnetic Laplacian for PERIODIC hierarchy, trophic
    Laplacian for LINEAR hierarchy (MacKay/Johnson trophic levels; Fanuel et al.
    magnetic eigenmaps). A layered codebase is linear: controller -> service ->
    repository. So compute TROPHIC LEVELS on the typed digraph and let the phase
    encode height DIFFERENCE:

        A_uv = w_uv * exp(i * 2*pi*g * (h_v - h_u))

    The phase now comes from the graph, lives on a circle by construction, and
    means something: how far up the architecture an edge climbs. Trophic levels
    solve the Laplacian system (D_in + D_out - A - A^T) h = D_in - D_out.

FIX 3 — the alternating loop (fibers <-> partition).
    Fibers indicate; the partition corrects; better fibers result. Round r:
      1. cluster the fiber graph -> seeds
      2. propagate over content-kNN -> node partition
      3. score each fiber by how concentrated its members are in one part
         (purity); RE-WEIGHT fibers by that purity
      4. recluster the re-weighted fiber graph
    Fibers whose members scatter across parts get demoted; fibers that predict
    the partition get amplified. Two to three rounds; every round measured, so a
    round that stops helping is visible rather than assumed.

Registered bar unchanged: >=18/20 paired wins on held-out co-change modularity
against the champion. Reported per round, not just at the end.
"""
import collections
import itertools
import math
import re
import subprocess

import numpy as np
import networkx as nx
import scipy.sparse as sp
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
KNN, N_SPLITS, SEED = 12, 20, 42
ROUNDS = 3
G_PHASE = 0.25
TARGET = (16, 22)

_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def idents(t):
    for w in _TOK.findall(t):
        w = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w)
        for p in re.split(r"[_\s]+", w):
            if len(p) > 2:
                yield p.lower()


def trophic_levels(n, edges):
    """MacKay/Johnson trophic levels: (D_in + D_out - A - A^T) h = D_in - D_out."""
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] += 1.0
    din, dout = A.sum(axis=0), A.sum(axis=1)
    L = np.diag(din + dout) - A - A.T
    b = din - dout
    h = np.zeros(n)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((u, v) for u, v in edges if u != v)
    for comp in nx.connected_components(G):
        c = sorted(comp)
        if len(c) < 2:
            continue
        sub = L[np.ix_(c, c)] + 1e-9 * np.eye(len(c))
        try:
            hc = np.linalg.lstsq(sub, b[c], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        hc -= hc.min()
        h[c] = hc
    return h


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
                "RETURN h.metapath AS mp, h.member_ids AS members, "
                "h.idf_weight AS idf, h.hub_id AS hub", ns=NS))
            tedges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "WHERE type(r) IN ['PERFORMS','USES','MODIFIES','CALLS','ACCESSES'] "
                "RETURN DISTINCT id(a) AS s, id(b) AS t", ns=NS))
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
    E = [(idx[e["s"]], idx[e["t"]]) for e in tedges
         if e["s"] in idx and e["t"] in idx and idx[e["s"]] != idx[e["t"]]]

    # ---------- FIX 2: trophic levels as a graph-native height
    h = trophic_levels(n, E)
    print("FIX 2 — TROPHIC HEIGHT (graph-native phase, not fitted)\n")
    for t in ("Actor", "Process", "Resource", "Rule", "Context"):
        m = et == t
        if m.sum() and h[m].any():
            print(f"  {t:<10} n={int(m.sum()):>4}  mean height {h[m].mean():>6.3f}")
    print("  a linear hierarchy should order Actor < Process < Resource;")
    print("  the literature's split: magnetic Laplacian = periodic hierarchy,")
    print("  trophic Laplacian = linear hierarchy, and a layered codebase is linear.\n")

    # magnetic Laplacian with the trophic phase, spectral embedding
    Aw = np.zeros((n, n), dtype=complex)
    for u, v in E:
        ph = np.exp(1j * 2 * np.pi * G_PHASE * (h[v] - h[u]))
        Aw[u, v] += ph
        Aw[v, u] += np.conj(ph)
    deg = np.abs(Aw).sum(axis=1)
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    Lm = np.eye(n, dtype=complex) - (inv[:, None] * Aw * inv[None, :])
    w, V = np.linalg.eigh(Lm)
    MAG = np.hstack([V[:, :8].real, V[:, :8].imag])
    MAG /= np.maximum(np.linalg.norm(MAG, axis=1, keepdims=True), 1e-12)

    # ---------- content kNN base (champion)
    Sq = X @ X.T
    np.fill_diagonal(Sq, -np.inf)
    W0 = np.zeros((n, n))
    knn_lists = {}
    for i in range(n):
        row = Sq[i].copy()
        row[[j for j in range(n) if repo_of[j] != repo_of[i]]] = -np.inf
        top = [j for j in np.argpartition(-row, KNN)[:KNN] if np.isfinite(row[j]) and row[j] > 0]
        knn_lists[i] = top
        for j in top:
            W0[i, j] = W0[j, i] = max(W0[i, j], row[j])

    # ---------- fibers
    H = []
    for k, hy in enumerate(hyper):
        mem = [idx[m] for m in hy["members"] if m in idx]
        if hy["hub"] in idx:
            mem.append(idx[hy["hub"]])
        if len(mem) >= 2:
            H.append([k, sorted(set(mem)), max(hy["idf"], 0.05)])

    def fiber_graph(weights):
        FG = nx.Graph()
        FG.add_nodes_from(range(len(H)))
        for (i, (_, mi, _)), (j, (_, mj, _)) in itertools.combinations(enumerate(H), 2):
            sh = len(set(mi) & set(mj))
            if sh:
                FG.add_edge(i, j, weight=sh * (weights[i] + weights[j]) / 2.0)
        return FG

    def seeds_from(FG, res):
        comms = nx.community.louvain_communities(FG, weight="weight",
                                                 resolution=res, seed=SEED)
        fcl = {f: c for c, grp in enumerate(comms) for f in grp}
        score = collections.defaultdict(lambda: collections.defaultdict(float))
        for i, (_, mem, w) in enumerate(H):
            for m in mem:
                score[m][fcl[i]] += w
        return {m: max(sc, key=sc.get) for m, sc in score.items()}, len(comms)

    # ---------- FIX 1: label propagation over content-kNN from fiber seeds
    def propagate(seed, iters=30):
        lab = np.full(n, -1)
        for m, c in seed.items():
            lab[m] = c
        for _ in range(iters):
            changed = 0
            order = np.random.default_rng(SEED).permutation(n)
            for i in order:
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

    # ---------- evaluation machinery
    by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}
    raw = {}
    for rp in REPOS:
        out = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        raw[rp] = [ids for ids in
                   (sorted({by_repo[rp][f] for f in
                            {l.strip() for l in b.splitlines() if l.strip()}
                            if f in by_repo[rp]}) for b in out.split("\x01")[1:])
                   if 2 <= len(ids) <= 30]
    pairs = [(a, b) for a, b in itertools.combinations(range(n), 2)
             if repo_of[a] == repo_of[b]]
    P = np.array(pairs)

    def split_y(rng, half):
        lab = collections.Counter()
        for rp in REPOS:
            keep = rng.random(len(raw[rp])) < 0.5
            for ids, k in zip(raw[rp], keep):
                if k != half:
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

    def type_entropy(lab):
        tot = wsum = 0.0
        for c in set(lab):
            m = lab == c
            cnt = collections.Counter(et[m])
            k = m.sum()
            tot += -sum((v / k) * math.log2(v / k) for v in cnt.values() if v) * k
            wsum += k
        return tot / max(wsum, 1)

    champ = None
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for a, b in np.argwhere(np.triu(W0, 1) > 0):
        g.add_edge(int(a), int(b), weight=float(W0[a, b]))
    for res in (1.18,):
        cm = nx.community.louvain_communities(g, weight="weight", resolution=res, seed=SEED)
        champ = np.full(n, -1)
        for c, grp in enumerate(cm):
            for v in grp:
                champ[v] = c

    # ---------- FIX 3: the alternating loop
    print("FIX 1+3 — LABEL PROPAGATION FROM FIBER SEEDS, THEN ALTERNATION\n")
    weights = [w for _, _, w in H]
    arms = {"A0 champion (control)": champ}
    for r in range(1, ROUNDS + 1):
        res = 0.5
        seed, nf = seeds_from(fiber_graph(weights), res)
        lab = propagate(seed)
        k = len(set(lab))
        arms[f"R{r} fiber->propagate"] = lab.copy()
        # re-weight fibers by partition purity
        neww = []
        for i, (_, mem, w0) in enumerate(H):
            cnt = collections.Counter(lab[m] for m in mem)
            purity = cnt.most_common(1)[0][1] / len(mem)
            neww.append(max(0.05, w0 * purity))
        drift = float(np.mean([abs(a - b) for a, b in zip(neww, weights)]))
        print(f"  round {r}: {nf} fiber-clusters, {len(seed)} seeded, {k} parts, "
              f"type-H {type_entropy(lab):.3f}, weight drift {drift:.4f}")
        weights = neww

    # magnetic-trophic arm: cluster the magnetic embedding, same pipeline shape
    Sm = MAG @ MAG.T
    np.fill_diagonal(Sm, -np.inf)
    Wm = np.zeros((n, n))
    for i in range(n):
        row = Sm[i].copy()
        row[[j for j in range(n) if repo_of[j] != repo_of[i]]] = -np.inf
        for j in np.argpartition(-row, KNN)[:KNN]:
            if np.isfinite(row[j]) and row[j] > 0:
                Wm[i, j] = Wm[j, i] = max(Wm[i, j], row[j])
    gm = nx.Graph()
    gm.add_nodes_from(range(n))
    for a, b in np.argwhere(np.triu(Wm, 1) > 0):
        gm.add_edge(int(a), int(b), weight=float(Wm[a, b]))
    lo, hi = 0.2, 8.0
    for _ in range(12):
        mid = (lo + hi) / 2
        cm = nx.community.louvain_communities(gm, weight="weight", resolution=mid, seed=SEED)
        mlab = np.full(n, -1)
        for c, grp in enumerate(cm):
            for v in grp:
                mlab[v] = c
        if TARGET[0] <= len(cm) <= TARGET[1]:
            break
        lo, hi = (mid, hi) if len(cm) < TARGET[0] else (lo, mid)
    arms["MAG trophic-phase kNN"] = mlab

    print("\nHELD-OUT BATTERY — 20 splits, paired vs champion\n")
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in range(N_SPLITS):
        y = split_y(np.random.default_rng(SEED + t), False)
        for nm, lab in arms.items():
            acc[nm]["mod"].append(modularity(lab, y))
            acc[nm]["coh"].append(coherence(lab, y))
    print(f"{'arm':<30}{'parts':>7}{'coherence':>11}{'modularity':>12}{'type-H':>9}{'wins':>8}")
    base = np.array(acc["A0 champion (control)"]["mod"])
    for nm, lab in arms.items():
        m = np.array(acc[nm]["mod"])
        d = m - base
        wins = "—" if nm.startswith("A0") else f"{int((d > 0).sum())}/{N_SPLITS}"
        print(f"{nm:<30}{len(set(lab)):>7}{np.nanmean(acc[nm]['coh']):>10.2f}x"
              f"{m.mean():>12.4f}{type_entropy(lab):>9.3f}{wins:>8}")
    print("\n  bar: >=18/20. type-H ceiling 2.027 (higher = more vertical/slice-like)")


if __name__ == "__main__":
    main()
