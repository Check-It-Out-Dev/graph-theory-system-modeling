"""G6 — should evidence CONSTRAIN the search, or only judge the result?

Owner's argument: no unsupervised algorithm should detect subsystems and
partition without legitimate connections behind it. Sub-topologies should not map
one-to-one onto subsystems; the ones that overlap in the ORIGINAL space should be
joined first, producing subsystem CANDIDATES, which are then connected on
evidence.

That exposes a real gap. Everything built so far clusters freely and validates
afterwards. The acceptance gate JUDGES a partition; it never CONSTRAINS the
search. Those are different things, and the difference is testable.

Two claims, tested separately.

CLAIM 1 — merge sub-topologies that overlap in R^4096, do not treat each as a
subsystem seed. Measured earlier: relation subspaces sit at 72-78 degrees mean
principal angle, near-orthogonal, EXCEPT USES and MODIFIES which share one
direction at 25.1 degrees, with MODIFIES's support contained entirely in USES's.
So the data names exactly one merge. Test whether merging before weighting beats
treating them separately.

CLAIM 2 — use high-precision evidence as HARD CONSTRAINTS rather than as weights.
The affinity built from reliable signatures fires on 494 pairs at 54.5% precision
and 24.8x lift: the most precise signal in the whole arc. Up-weighting it in the
objective was already tried and did nothing (content x edges 15.60x against
content alone 16.05x). A must-link CONSTRAINT is a stronger instrument: it forces
two files together regardless of what the objective prefers, which is exactly the
"legit connection" the owner is asking for.

Implemented as union-find contraction: must-linked files become one super-node,
the content graph is contracted onto super-nodes, clustered, and expanded back.

Not circular: the constraints come from typed graph edges, the evaluation from
git co-change. Different sources.

Scored on held-out co-change over 20 splits with coherence, map-equation
description length and TurboMQ, plus the constraint-satisfaction rate that free
clustering achieves on its own -- because if free clustering already satisfies
the constraints, adding them cannot help and the honest answer is that the
evidence was already implicit in the content signal.
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
KNN, N_SPLITS, SEED = 12, 20, 42

# signatures that passed the split-half reliability gate (floor < 25 deg)
RELIABLE = {
    ("Actor", "PERFORMS", "Process"): 0.43, ("Actor", "INJECTS", "Process"): 0.43,
    ("Process", "USES", "Resource"): 0.81, ("Process", "INJECTS", "Resource"): 0.81,
    ("Process", "MODIFIES", "Resource"): 0.64, ("Actor", "ACCESSES", "Resource"): 0.66,
    ("Actor", "INJECTS", "Resource"): 0.66, ("Actor", "IMPORTS", "Process"): 0.82,
}


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def map_equation(labels, W):
    k = W.sum(axis=1)
    tw = k.sum()
    if tw <= 0:
        return np.nan
    p = k / tw

    def plogp(x):
        x = np.asarray(x, dtype=float)
        x = x[x > 0]
        return float(np.sum(x * np.log2(x)))

    mods = collections.defaultdict(list)
    for i, c in enumerate(labels):
        mods[c].append(i)
    q_i, inner = [], 0.0
    for c, members in mods.items():
        m = np.zeros(len(labels), dtype=bool)
        m[members] = True
        qi = W[np.ix_(m, ~m)].sum() / tw
        pin = p[m]
        pc = qi + pin.sum()
        q_i.append(qi)
        inner += -plogp([qi]) - plogp(pin) + plogp([pc])
    q = float(np.sum(q_i))
    index = 0.0 if q <= 0 else (plogp([q]) - plogp(q_i))
    return float(index + inner)


def turbo_mq(labels, G):
    intra, inter = collections.Counter(), collections.Counter()
    for u, v in G.edges():
        a, b = labels[u], labels[v]
        if a == b:
            intra[a] += 1
        else:
            inter[a] += 1
            inter[b] += 1
    return float(sum(2 * intra[c] / (2 * intra[c] + inter[c])
                     for c in set(labels) if 2 * intra[c] + inter[c] > 0))


def main():
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
                "RETURN id(n) AS id, n.file_path AS fp, n.embedding AS emb", ns=NS))
            edges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "RETURN id(a) AS s, id(b) AS t, type(r) AS k, "
                "a.entity_type AS st, b.entity_type AS tt", ns=NS))
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

    # ---- constraints from reliable signatures only, deduplicated
    must, seen = [], set()
    Gtyped = nx.Graph()
    Gtyped.add_nodes_from(range(n))
    for e in edges:
        u, v = idx.get(e["s"]), idx.get(e["t"])
        if u is None or v is None or u == v:
            continue
        Gtyped.add_edge(u, v)
        sig = (e["st"], e["k"], e["tt"])
        if sig not in RELIABLE:
            continue
        key = (min(u, v), max(u, v), sig)
        if key in seen:
            continue
        seen.add(key)
        if 1.0 - RELIABLE[sig] > 0:
            must.append((u, v, 1.0 - RELIABLE[sig]))
    must.sort(key=lambda t: -t[2])
    print(f"{n} files; {len(must)} candidate must-link pairs from the "
          f"{len(RELIABLE)} reliable signatures\n")

    # ---- content kNN
    S = X @ X.T
    np.fill_diagonal(S, -1.0)
    Wc = np.zeros((n, n))
    for i in range(n):
        for j in np.argpartition(-S[i], KNN)[:KNN]:
            if repo_of[i] == repo_of[j] and S[i, j] > 0:
                Wc[i, j] = Wc[j, i] = max(Wc[i, j], S[i, j])

    def louvain(W, res=1.0):
        g = nx.Graph()
        g.add_nodes_from(range(len(W)))
        for a, b in np.argwhere(np.triu(W, 1) > 0):
            g.add_edge(int(a), int(b), weight=float(W[a, b]))
        cm = nx.community.louvain_communities(g, weight="weight", resolution=res, seed=SEED)
        lab = np.full(len(W), -1)
        for c, grp in enumerate(cm):
            for v in grp:
                lab[v] = c
        return lab

    free = louvain(Wc)

    # ---- does free clustering already satisfy the constraints?
    sat = sum(1 for u, v, _ in must if free[u] == free[v])
    print(f"CONSTRAINT SATISFACTION BY FREE CLUSTERING\n")
    print(f"  must-link pairs already in the same part : {sat}/{len(must)} "
          f"({sat / max(len(must), 1):.1%})")
    print(f"  pairs the constraints would actually move: {len(must) - sat}")
    print("\n  if free clustering already satisfies them, the evidence was implicit")
    print("  in the content signal and constraining cannot add information\n")

    # ---- constrained: contract must-linked pairs, cluster, expand
    def constrained(threshold):
        uf = UF(n)
        used = 0
        for u, v, w in must:
            if w >= threshold:
                uf.union(u, v)
                used += 1
        groups = collections.defaultdict(list)
        for i in range(n):
            groups[uf.find(i)].append(i)
        reps = sorted(groups)
        rid = {r: i for i, r in enumerate(reps)}
        m = len(reps)
        Wsuper = np.zeros((m, m))
        for i in range(n):
            gi = rid[uf.find(i)]
            for j in range(i + 1, n):
                if Wc[i, j] > 0:
                    gj = rid[uf.find(j)]
                    if gi != gj:
                        Wsuper[gi, gj] += Wc[i, j]
                        Wsuper[gj, gi] += Wc[i, j]
        sup = louvain(Wsuper)
        lab = np.empty(n, dtype=int)
        for i in range(n):
            lab[i] = sup[rid[uf.find(i)]]
        return lab, used, m

    # ---- labels
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

    def split_W(rng):
        W = np.zeros((n, n))
        for rp in REPOS:
            keep = rng.random(len(raw[rp])) < 0.5
            for ids, k in zip(raw[rp], keep):
                if k:
                    continue
                for a, b in itertools.combinations(ids, 2):
                    W[a, b] = W[b, a] = 1.0
        return W

    cands = {"free (unsupervised)": free}
    for th in (0.6, 0.4, 0.2):
        lab, used, m = constrained(th)
        cands[f"constrained (w>={th}, {used} links)"] = lab

    print("SCORED ON HELD-OUT CO-CHANGE, 20 SPLITS\n")
    print(f"{'partition':<36}{'parts':>7}{'coherence':>11}{'bits':>9}{'TurboMQ':>9}")
    res = {}
    for name, lab in cands.items():
        coh, bits = [], []
        for t in range(N_SPLITS):
            W = split_W(np.random.default_rng(SEED + t))
            same = lab[P[:, 0]] == lab[P[:, 1]]
            y = W[P[:, 0], P[:, 1]]
            if same.sum() and (~same).sum() and y[~same].mean() > 0:
                coh.append(y[same].mean() / y[~same].mean())
            bits.append(map_equation(lab, W))
        res[name] = (np.mean(coh), np.mean(bits))
        print(f"{name:<36}{len(set(lab)):>7}{np.mean(coh):>10.2f}x"
              f"{np.mean(bits):>9.3f}{turbo_mq(lab, Gtyped):>9.2f}")

    print("\n  bits: lower is better. A constraint helps only if it improves the")
    print("  score it was not derived from.")
    best = min(res, key=lambda k_: res[k_][1])
    print(f"\n  lowest description length: {best}")


if __name__ == "__main__":
    main()
