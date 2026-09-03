"""S3b — the granularity confound, and the comparison that survives it.

S3 reported directory structure at 9.89x against a derived partition at 6.99x and
concluded directory wins. That comparison is rigged, and I built the rig:
directory had 248 parts, the derived partition had 18. Coherence rises mechanically
as parts get smaller -- a partition into pairs would score enormously and mean
nothing -- so any comparison at unequal part counts measures granularity, not
quality.

The fix is a curve, not a number. Sweep both methods across part counts and
compare at matched granularity:

  derived     Louvain resolution from 0.2 to 20, which trades off part count
  directory   truncate each path to its first d components, d = 1..8, which gives
              the natural coarse-to-fine family of the folder tree

At the same number of parts, whichever curve is higher is genuinely better. And
the honest failure mode to watch for: if directory dominates at every granularity,
the geometry is not earning its complexity and should be reported as such.

Held out by commit, as before.
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
MAX_COMMIT_FILES, KNN, SEED = 30, 12, 42

IDENTITY_ERR = {
    ("Actor", "PERFORMS", "Process"): 0.43, ("Actor", "INJECTS", "Process"): 0.43,
    ("Actor", "INJECTS", "Resource"): 0.66, ("Actor", "ACCESSES", "Resource"): 0.66,
    ("Process", "MODIFIES", "Resource"): 0.64, ("Process", "USES", "Resource"): 0.81,
    ("Process", "INJECTS", "Resource"): 0.81, ("Process", "INJECTS", "Process"): 0.82,
    ("Process", "CALLS", "Process"): 0.82, ("Actor", "IMPORTS", "Process"): 0.82,
}


def main():
    rng = np.random.default_rng(SEED)
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run("MATCH (n:EntityDetail {namespace:$ns}) "
                               "WHERE n.embedding IS NOT NULL RETURN id(n) AS id, "
                               "n.file_path AS fp, n.embedding AS emb", ns=NS))
            edges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "RETURN id(a) AS s, id(b) AS t, type(r) AS k, a.entity_type AS st, "
                "b.entity_type AS tt", ns=NS))
    finally:
        drv.close()

    idx = {r["id"]: i for i, r in enumerate(nodes)}
    n = len(nodes)
    repo_of, rel_of = {}, {}
    for i, r in enumerate(nodes):
        rest = (r["fp"] or "").replace("\\", "/")
        rest = rest[len(ROOT):] if rest.startswith(ROOT) else rest
        p = rest.split("/", 1)
        repo_of[i], rel_of[i] = (p[0], p[1]) if len(p) == 2 else ("?", rest)
    X = np.array([r["emb"] for r in nodes], dtype=np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}
    tr, te = collections.Counter(), collections.Counter()
    for rp in REPOS:
        out = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        sets = [sorted({l.strip() for l in b.splitlines() if l.strip()})
                for b in out.split("\x01")[1:]]
        sets = [s for s in sets if 2 <= len(s) <= MAX_COMMIT_FILES]
        m = rng.random(len(sets)) < 0.5
        for grp, tgt in ((np.array(sets, dtype=object)[m], tr),
                         (np.array(sets, dtype=object)[~m], te)):
            for files in grp:
                ids = sorted({by_repo[rp][f] for f in files if f in by_repo[rp]})
                for a, b in itertools.combinations(ids, 2):
                    tgt[(a, b)] += 1

    pairs = [(a, b) for a, b in itertools.combinations(range(n), 2)
             if repo_of[a] == repo_of[b]]
    P = np.array(pairs)
    y_te = np.array([1 if te.get((a, b), 0) else 0 for a, b in pairs])

    def coh(part):
        same = part[P[:, 0]] == part[P[:, 1]]
        if same.sum() == 0 or (~same).sum() == 0:
            return np.nan
        return float(y_te[same].mean() / max(y_te[~same].mean(), 1e-12))

    S = X @ X.T
    np.fill_diagonal(S, -1.0)
    W = np.zeros((n, n))
    for i in range(n):
        for j in np.argpartition(-S[i], KNN)[:KNN]:
            if repo_of[i] == repo_of[j] and S[i, j] > 0:
                W[i, j] = W[j, i] = max(W[i, j], S[i, j])

    Eb, seen = np.zeros((n, n)), set()
    for e in edges:
        u, v = idx.get(e["s"]), idx.get(e["t"])
        if u is None or v is None or u == v:
            continue
        key = (min(u, v), max(u, v), e["st"], e["k"], e["tt"])
        if key in seen:
            continue
        seen.add(key)
        w = max(0.0, 1.0 - IDENTITY_ERR.get((e["st"], e["k"], e["tt"]), 1.0))
        Eb[u, v] = Eb[v, u] = max(Eb[u, v], w)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    Wb = W * (1.0 + 2.0 * Eb)
    for a, b in np.argwhere(np.triu(Wb, 1) > 0):
        G.add_edge(int(a), int(b), weight=float(Wb[a, b]))
    Gc = nx.Graph()
    Gc.add_nodes_from(range(n))
    for a, b in np.argwhere(np.triu(W, 1) > 0):
        Gc.add_edge(int(a), int(b), weight=float(W[a, b]))

    def louvain(g, res):
        comms = nx.community.louvain_communities(g, weight="weight",
                                                 resolution=res, seed=SEED)
        p = np.full(n, -1)
        for c, grp in enumerate(comms):
            for v in grp:
                p[v] = c
        return p

    print("COHERENCE vs GRANULARITY, held out by commit\n")
    print("  coherence rises mechanically as parts shrink, so only equal part")
    print("  counts may be compared. Read down to matching `parts`.\n")

    rows = []
    for res in [0.2, 0.5, 1, 2, 4, 8, 16, 32, 64]:
        for name, g in (("content only", Gc), ("content x edges", Gb := G)):
            p = louvain(g, res)
            rows.append((name, len(set(p)), coh(p), res))

    dirrows = []
    for depth in range(1, 9):
        lab = np.array([hash(repo_of[i] + "/" + "/".join(
            rel_of[i].split("/")[:depth])) % 10 ** 9 for i in range(n)])
        dirrows.append(("directory", len(set(lab)), coh(lab), depth))

    print(f"{'method':<20}{'parts':>7}{'coherence':>12}   knob")
    for name, k, c, knob in sorted(dirrows, key=lambda r: r[1]):
        print(f"{name:<20}{k:>7}{c:>11.2f}x   depth {knob}")
    print()
    for name, k, c, knob in sorted([r for r in rows if r[0] == "content only"],
                                   key=lambda r: r[1]):
        print(f"{name:<20}{k:>7}{c:>11.2f}x   res {knob}")
    print()
    for name, k, c, knob in sorted([r for r in rows if r[0] == "content x edges"],
                                   key=lambda r: r[1]):
        print(f"{name:<20}{k:>7}{c:>11.2f}x   res {knob}")

    print("\n\nMATCHED COMPARISON — nearest directory depth by part count\n")
    print(f"{'parts (derived)':>16}{'derived':>11}{'nearest dir':>13}{'dir parts':>11}   verdict")
    for name, k, c, knob in sorted([r for r in rows if r[0] == "content x edges"],
                                   key=lambda r: r[1]):
        d = min(dirrows, key=lambda r: abs(r[1] - k))
        if np.isnan(c) or np.isnan(d[2]):
            continue
        verdict = "derived wins" if c > d[2] else "directory wins"
        print(f"{k:>16}{c:>10.2f}x{d[2]:>12.2f}x{d[1]:>11}   {verdict}")


if __name__ == "__main__":
    main()
