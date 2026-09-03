"""#42 — the fused partition: RRF affinity + IDF-weighted hyperedges + layer view.

The owner's proposal, mapped onto what is measured to exist:

  "triple embeddings"      -> the ranker of record, RRF(content, lexical) = 0.8432,
                              which the PARTITIONER never received - the champion
                              still clusters content-only kNN. Upgrade it.
  "paths in sub-topologies
   crossing into others"   -> the 92 IDF-weighted hyperedges: A_P_R is literally an
                              Actor->Process->Resource path bundled as one object,
                              within-cohort co-change 68.4% vs 3.15% base.
  "the layer view"         -> entity_type as DIAGNOSTIC, not signal: a subsystem is
                              a vertical slice (F64: 86% of entropy ceiling), so a
                              layer-pure cluster is a layer, not a subsystem.
                              Report entropy per arm; do not fuse layers into the
                              objective.

Arms, cumulative so each step's contribution is isolated:

  A0  champion control — content-only kNN, Louvain          (the 18-subsystem winner)
  A1  RRF(content, lexical) kNN, Louvain                    (ranker upgrade only)
  A2  A1 + hyperedge clique-expansion, IDF-weighted, λ*     (the fusion)

Honest priors, stated before running: A2 may fail. Soft-boosting raw typed edges
failed (F27, 15.60x vs 16.05x), and 64.8% of typed must-links were already
satisfied by free clustering (F59). What is new here: the boost objects are
n-ary, IDF-cleaned, at 68.4% weighted precision instead of 54.5%, on a better
base graph. The experiment decides; 0/20 would close the reinforcement idea for
good, >=18/20 promotes it.

Controls per the V4 prompt: λ chosen on TRAIN halves only; granularity matched by
tuning each arm's Louvain resolution to the champion's part count; 20 independent
commit splits; paired win counts; degenerate-partition sanity implicitly via the
battery (map equation reports vs single-part).
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
RRF_K = 60.0
LAMBDAS = [0.5, 1.0, 2.0, 4.0]
TARGET_PARTS = (16, 22)          # champion granularity band

_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def idents(t):
    for w in _TOK.findall(t):
        w = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w)
        for p in re.split(r"[_\s]+", w):
            if len(p) > 2:
                yield p.lower()


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

    bags = []
    for r in nodes:
        try:
            bags.append(collections.Counter(
                idents(open(r["fp"], encoding="utf-8", errors="replace").read()[:24_000])))
        except Exception:
            bags.append(collections.Counter())
    df = collections.Counter()
    for b in bags:
        df.update(b.keys())
    vocab = {w: i for i, (w, _) in enumerate(df.most_common(20_000))}
    rows, cols, vals = [], [], []
    for i, b in enumerate(bags):
        for w, c in b.items():
            j = vocab.get(w)
            if j is not None:
                rows.append(i); cols.append(j)
                vals.append((1 + math.log(c)) * math.log(n / df[w]))
    Tf = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)))
    nr = np.sqrt(Tf.multiply(Tf).sum(axis=1)).A.ravel()
    nr[nr == 0] = 1.0
    Td = np.asarray((sp.diags(1 / nr) @ Tf).todense())

    Sq = X @ X.T
    St = Td @ Td.T
    np.fill_diagonal(Sq, -np.inf)
    np.fill_diagonal(St, -np.inf)

    def knn_graph(fused):
        W = np.zeros((n, n))
        for i in range(n):
            row = fused[i].copy()
            row[[j for j in range(n) if repo_of[j] != repo_of[i]]] = -np.inf
            for j in np.argpartition(-row, KNN)[:KNN]:
                if np.isfinite(row[j]) and row[j] > 0:
                    W[i, j] = W[j, i] = max(W[i, j], row[j])
        return W

    def rrf_rows(A, B):
        F = np.zeros((n, n))
        for i in range(n):
            ra = np.argsort(np.argsort(-A[i]))
            rb = np.argsort(np.argsort(-B[i]))
            F[i] = 1.0 / (RRF_K + ra + 1) + 1.0 / (RRF_K + rb + 1)
        return F

    W0 = knn_graph(Sq)                       # champion base
    W1 = knn_graph(rrf_rows(Sq, St))         # RRF base

    # hyperedge pair boosts (clique expansion, IDF-weighted)
    boosts = collections.Counter()
    max_idf = max((h["idf"] for h in hyper), default=1.0) or 1.0
    for h in hyper:
        members = [idx[m] for m in h["members"] if m in idx]
        w = h["idf"] / max_idf
        if w <= 0 or len(members) < 2:
            continue
        for a, b in itertools.combinations(sorted(members), 2):
            if repo_of[a] == repo_of[b]:
                boosts[(a, b)] = max(boosts[(a, b)], w)
    med1 = np.median(W1[W1 > 0])

    def with_hyper(W, lam):
        W2 = W.copy()
        for (a, b), w in boosts.items():
            W2[a, b] = W2[b, a] = W2[a, b] + lam * med1 * w
        return W2

    def louvain_at(W, lo=0.2, hi=8.0):
        """Louvain with resolution tuned into the champion granularity band."""
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for a, b in np.argwhere(np.triu(W, 1) > 0):
            g.add_edge(int(a), int(b), weight=float(W[a, b]))
        best = None
        for _ in range(14):
            mid = (lo + hi) / 2
            cm = nx.community.louvain_communities(g, weight="weight",
                                                  resolution=mid, seed=SEED)
            k = len(cm)
            lab = np.full(n, -1)
            for c, grp in enumerate(cm):
                for v in grp:
                    lab[v] = c
            best = lab
            if TARGET_PARTS[0] <= k <= TARGET_PARTS[1]:
                return lab, k, mid
            if k < TARGET_PARTS[0]:
                lo = mid
            else:
                hi = mid
        return best, len(set(best)), (lo + hi) / 2

    # ---- co-change machinery
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

    def coherence(lab, y):
        same = lab[P[:, 0]] == lab[P[:, 1]]
        if not same.sum() or not (~same).sum() or y[~same].mean() == 0:
            return np.nan
        return float(y[same].mean() / y[~same].mean())

    def modularity(lab, y):
        W = np.zeros((n, n))
        W[P[:, 0], P[:, 1]] = y
        W[P[:, 1], P[:, 0]] = y
        k = W.sum(axis=1)
        m2 = W.sum()
        q = 0.0
        for c in set(lab):
            m = lab == c
            q += W[np.ix_(m, m)].sum() / m2 - (k[m].sum() / m2) ** 2
        return float(q)

    def map_eq(lab, y):
        W = np.zeros((n, n))
        W[P[:, 0], P[:, 1]] = y
        W[P[:, 1], P[:, 0]] = y
        k = W.sum(axis=1)
        tw = k.sum()
        if tw <= 0:
            return np.nan
        p = k / tw

        def plogp(x):
            x = np.asarray(x, float)
            x = x[x > 0]
            return float(np.sum(x * np.log2(x)))

        mods = collections.defaultdict(list)
        for i, c in enumerate(lab):
            mods[c].append(i)
        q_i, inner = [], 0.0
        for c, mem in mods.items():
            m = np.zeros(n, bool)
            m[mem] = True
            qi = W[np.ix_(m, ~m)].sum() / tw
            pin = p[m]
            q_i.append(qi)
            inner += -plogp([qi]) - plogp(pin) + plogp([qi + pin.sum()])
        q = float(np.sum(q_i))
        return float((0.0 if q <= 0 else (plogp([q]) - plogp(q_i))) + inner)

    Gt = nx.Graph()
    Gt.add_nodes_from(range(n))
    for e in tedges:
        u, v = idx.get(e["s"]), idx.get(e["t"])
        if u is not None and v is not None and u != v:
            Gt.add_edge(u, v)

    def turbo_mq(lab):
        intra, inter = collections.Counter(), collections.Counter()
        for u, v in Gt.edges():
            a, b = lab[u], lab[v]
            if a == b:
                intra[a] += 1
            else:
                inter[a] += 1
                inter[b] += 1
        return float(sum(2 * intra[c] / (2 * intra[c] + inter[c])
                         for c in set(lab) if 2 * intra[c] + inter[c] > 0))

    def type_entropy(lab):
        tot = wsum = 0.0
        for c in set(lab):
            m = lab == c
            cnt = collections.Counter(et[m])
            k = m.sum()
            h = -sum((v / k) * math.log2(v / k) for v in cnt.values() if v)
            tot += h * k
            wsum += k
        return tot / max(wsum, 1)

    # ---- choose lambda on TRAIN halves of 3 splits
    print("choosing lambda on train halves (coherence)\n")
    lam_score = {}
    for lam in LAMBDAS:
        lab, k, _ = louvain_at(with_hyper(W1, lam))
        cs = [coherence(lab, split_y(np.random.default_rng(SEED + t), True))
              for t in range(3)]
        lam_score[lam] = float(np.nanmean(cs))
        print(f"  lambda {lam:>4}: {k} parts, train coherence {lam_score[lam]:.2f}x")
    lam_star = max(lam_score, key=lam_score.get)
    print(f"  chosen lambda* = {lam_star}\n")

    # A3 closes the design hole: the boost must also be tested on the CHAMPION
    # base, else "does reinforcement help the best partition" stays unmeasured.
    med0 = np.median(W0[W0 > 0])

    def with_hyper0(lam):
        W2 = W0.copy()
        for (a, b), w in boosts.items():
            W2[a, b] = W2[b, a] = W2[a, b] + lam * med0 * w
        return W2

    arms = {}
    for name, W in (("A0 champion (content kNN)", W0),
                    ("A1 RRF(content+lexical) kNN", W1),
                    (f"A2 A1 + hyperedges (lam={lam_star})", with_hyper(W1, lam_star)),
                    (f"A3 A0 + hyperedges (lam={lam_star})", with_hyper0(lam_star))):
        lab, k, res = louvain_at(W)
        arms[name] = lab
        print(f"  {name}: {k} parts (res {res:.2f}), largest "
              f"{max(collections.Counter(lab).values())}")

    # ---- A4: the fiber-first partition (owner's construction, made precise)
    #
    # The typed graph projects onto the type quiver; a FIBER over a quiver path
    # is the set of concrete paths realising it — exactly the meta-path
    # instances, materialised as hyperedges. Two fibers are homotopy-adjacent
    # when they share members (a member-swap is the elementary deformation), so
    # communities of the fiber graph are the discrete homotopy-like classes of
    # vertical slices. Cluster the FIBERS, then induce the node partition:
    # fiber-core nodes join their strongest fiber-cluster, peripheral nodes
    # attach by content similarity to cluster cores. Topology chooses the
    # skeleton; the embedding only fleshes it out.
    H = [(i, [idx[m] for m in h["members"] if m in idx] +
             ([idx[h["hub"]]] if h["hub"] in idx else []),
          max(h["idf"], 0.05)) for i, h in enumerate(hyper)]
    FG = nx.Graph()
    FG.add_nodes_from(i for i, _, _ in H)
    for (i, mi, wi), (j, mj, wj) in itertools.combinations(H, 2):
        shared = len(set(mi) & set(mj))
        if shared:
            FG.add_edge(i, j, weight=shared * (wi + wj) / 2.0)

    def fiber_partition(res):
        comms = nx.community.louvain_communities(FG, weight="weight",
                                                 resolution=res, seed=SEED)
        fcl = {}
        for c, grp in enumerate(comms):
            for f in grp:
                fcl[f] = c
        score = collections.defaultdict(lambda: collections.defaultdict(float))
        for fi, members, w in H:
            for m in members:
                score[m][fcl[fi]] += w
        lab = np.full(n, -1)
        for m, sc in score.items():
            lab[m] = max(sc, key=sc.get)
        # periphery: attach by mean of top-3 content cosines to cluster cores
        core = collections.defaultdict(list)
        for m, c in ((m, lab[m]) for m in range(n) if lab[m] >= 0):
            core[c].append(m)
        for m in range(n):
            if lab[m] >= 0:
                continue
            best, bs = -1, -np.inf
            for c, mem in core.items():
                peers = [x for x in mem if repo_of[x] == repo_of[m]]
                if not peers:
                    continue
                sims = np.sort(Sq[m, peers])[-3:]
                v = float(np.mean(sims))
                if v > bs:
                    bs, best = v, c
            lab[m] = best if best >= 0 else max(core, key=lambda c: len(core[c]))
        return lab, len(set(lab))

    fres, flab, fk = 1.0, None, 0
    for res in (0.5, 1.0, 2.0, 3.0, 5.0):
        cand, k = fiber_partition(res)
        flab, fk, fres = cand, k, res
        if k >= 8:
            break
    arms[f"A4 fiber-first (k={fk}, res={fres})"] = flab
    print(f"  A4 fiber-first: {fk} node-partition parts from "
          f"{FG.number_of_nodes()} fibers, {FG.number_of_edges()} fiber adjacencies")
    if not (TARGET_PARTS[0] <= fk <= TARGET_PARTS[1]):
        print(f"  NOTE: A4 granularity {fk} is outside the champion band "
              f"{TARGET_PARTS} — coherence/TurboMQ comparisons carry the "
              f"granularity caveat; the map equation does not.")

    print("\nHELD-OUT BATTERY — 20 test halves, paired\n")
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in range(N_SPLITS):
        y = split_y(np.random.default_rng(SEED + t), False)
        for name, lab in arms.items():
            acc[name]["coh"].append(coherence(lab, y))
            acc[name]["mod"].append(modularity(lab, y))
            acc[name]["bits"].append(map_eq(lab, y))

    print(f"{'arm':<34}{'coherence':>11}{'modularity':>12}{'bits-lo':>9}"
          f"{'TurboMQ':>9}{'type-H':>8}")
    for name, lab in arms.items():
        c = np.nanmean(acc[name]["coh"])
        m = np.nanmean(acc[name]["mod"])
        b = np.nanmean(acc[name]["bits"])
        print(f"{name:<34}{c:>10.2f}x{m:>12.4f}{b:>9.3f}"
              f"{turbo_mq(lab):>9.2f}{type_entropy(lab):>8.3f}")

    names = list(arms)
    pairs_to_test = [(names[1], names[0]), (names[2], names[1]), (names[2], names[0])]
    for extra in names[3:]:
        pairs_to_test.append((extra, names[0]))
    for a, b in pairs_to_test:
        d = np.array(acc[a]["mod"]) - np.array(acc[b]["mod"])
        print(f"\n  {a}  vs  {b}\n    modularity Δ {d.mean():+.4f} ± {d.std():.4f}, "
              f"wins {int((d > 0).sum())}/{N_SPLITS}")
    print("\n  criterion: an arm is promoted only on >=18/20 paired wins")
    print("  type-H: global ceiling 2.027 bits; higher = more slice-like (F64)")


if __name__ == "__main__":
    main()
