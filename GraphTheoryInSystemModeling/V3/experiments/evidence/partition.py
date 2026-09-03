"""S3 — derive a partition, and score it on commits it has never seen.

6e set the target as a number instead of an aesthetic. A partition is good when
files in the same part actually change together, and the bar is directory
structure: 9.51x coherence at full coverage. It also identified the ingredients
and the mistake to avoid --

  content embedding   AUC 0.843, full coverage, but blind to structure
  typed edges         14x conditional lift, precision 0.545, coverage 0.57%
  the mistake         emb + lambda*g. Adding a sparse high-precision indicator to
                      a dense ranker dilutes the indicator and barely moves the
                      ranker. They compose; they do not sum.

So: W = kNN(content cosine) * (1 + beta * edge weight). Content decides who is
even a candidate; a typed edge multiplies the pair up when one exists. Then
Louvain on W, which picks its own cluster count by modularity rather than letting
me choose the number that flatters the result.

HELD OUT BY COMMIT. Any beta chosen by looking at co-change is fitted to
co-change, so the commits are split in half at random: build W and pick beta on
the first half, report every number on the second. A partition scored on the
commits that built it is not a measurement.
"""
import collections
import itertools
import subprocess

import numpy as np
import networkx as nx
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS = "CheckItOutV3"
ROOT = "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
MAX_COMMIT_FILES = 30
KNN = 12
SEED = 42

IDENTITY_ERR = {
    ("Actor", "PERFORMS", "Process"): 0.43, ("Actor", "INJECTS", "Process"): 0.43,
    ("Actor", "INJECTS", "Resource"): 0.66, ("Actor", "ACCESSES", "Resource"): 0.66,
    ("Process", "MODIFIES", "Resource"): 0.64, ("Process", "USES", "Resource"): 0.81,
    ("Process", "INJECTS", "Resource"): 0.81, ("Process", "INJECTS", "Process"): 0.82,
    ("Process", "CALLS", "Process"): 0.82, ("Actor", "IMPORTS", "Process"): 0.82,
}


def commit_file_sets(repo):
    out = subprocess.run(
        ["git", "-C", ROOT + repo, "log", "--all", "--pretty=format:\x01", "--name-only"],
        capture_output=True, text=True, errors="replace").stdout
    sets = []
    for block in out.split("\x01")[1:]:
        files = sorted({l.strip() for l in block.splitlines() if l.strip()})
        if 2 <= len(files) <= MAX_COMMIT_FILES:
            sets.append(files)
    return sets


def labels_from(commit_sets, index_by_repo, repo):
    lab = collections.Counter()
    here = index_by_repo[repo]
    for files in commit_sets:
        ids = [here[f] for f in files if f in here]
        for a, b in itertools.combinations(sorted(set(ids)), 2):
            lab[(a, b)] += 1
    return lab


def coherence(part, pairs, y, repo_of):
    """P(co-change | same part) / P(co-change | different part)."""
    same = np.array([part[a] == part[b] for a, b in pairs])
    if same.sum() == 0 or (~same).sum() == 0:
        return np.nan, 0, 0
    return (y[same].mean() / max(y[~same].mean(), 1e-12),
            int(same.sum()), float(y[same].mean()))


def main():
    rng = np.random.default_rng(SEED)
    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with driver.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
                "RETURN id(n) AS id, n.file_path AS fp, n.subsystem_id AS sub, "
                "n.embedding AS emb", ns=NS))
            edges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "RETURN id(a) AS s, id(b) AS t, type(r) AS k, "
                "a.entity_type AS st, b.entity_type AS tt", ns=NS))
    finally:
        driver.close()

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
    sub = np.array([r["sub"] if r["sub"] is not None else -1 for r in nodes])
    dirs = np.array([rel_of[i].rsplit("/", 1)[0] if "/" in rel_of[i] else "" for i in range(n)])

    index_by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}

    # ---- split commits in half; train picks beta, test reports
    train_lab, test_lab = collections.Counter(), collections.Counter()
    ncom = 0
    for rp in REPOS:
        sets = commit_file_sets(rp)
        ncom += len(sets)
        mask = rng.random(len(sets)) < 0.5
        train_lab.update(labels_from([s for s, m in zip(sets, mask) if m], index_by_repo, rp))
        test_lab.update(labels_from([s for s, m in zip(sets, mask) if not m], index_by_repo, rp))
    print(f"{n} files, {ncom} usable commits, split into "
          f"{len(train_lab):,} train pairs / {len(test_lab):,} test pairs\n")

    pairs = [(a, b) for a, b in itertools.combinations(range(n), 2) if repo_of[a] == repo_of[b]]
    P = np.array(pairs)
    y_tr = np.array([1 if train_lab.get((a, b), 0) else 0 for a, b in pairs])
    y_te = np.array([1 if test_lab.get((a, b), 0) else 0 for a, b in pairs])
    print(f"within-repo pairs {len(pairs):,}; positives train {y_tr.sum():,} "
          f"test {y_te.sum():,}\n")

    # ---- content kNN graph
    S = X @ X.T
    np.fill_diagonal(S, -1.0)
    W0 = np.zeros((n, n))
    for i in range(n):
        for j in np.argpartition(-S[i], KNN)[:KNN]:
            if repo_of[i] == repo_of[j] and S[i, j] > 0:
                W0[i, j] = W0[j, i] = max(W0[i, j], S[i, j])

    # ---- typed-edge boost, deduplicated
    Eb = np.zeros((n, n))
    seen = set()
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

    def louvain(W):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        nz = np.argwhere(np.triu(W, 1) > 0)
        G.add_weighted_edges_from((int(a), int(b), float(W[a, b])) for a, b in nz)
        comms = nx.community.louvain_communities(G, weight="weight", seed=SEED)
        part = np.full(n, -1)
        for c, grp in enumerate(comms):
            for v in grp:
                part[v] = c
        return part

    # ---- pick beta on TRAIN only
    print("choosing the edge boost on training commits only\n")
    print(f"{'beta':>6}{'clusters':>10}{'train coherence':>18}")
    best = (None, -1)
    for beta in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        part = louvain(W0 * (1.0 + beta * Eb))
        c, _, _ = coherence(part, pairs, y_tr, repo_of)
        print(f"{beta:>6.1f}{len(set(part)):>10}{c:>17.2f}x")
        if c > best[1]:
            best = (beta, c)
    beta = best[0]
    print(f"\n  chosen beta = {beta:g}\n")

    # ---- final partition, reported on TEST
    part = louvain(W0 * (1.0 + beta * Eb))
    sizes = collections.Counter(part)
    covered = sum(v for v in sizes.values() if v > 1) / n

    print("HELD-OUT COHERENCE — every number below is on commits not used above\n")
    print(f"{'partition':<34}{'parts':>7}{'coverage':>10}{'coherence':>12}")
    cands = {
        "directory structure": np.array([hash(d) % 10 ** 9 for d in dirs]),
        "incumbent subsystem_id": sub,
        "content only (beta=0)": louvain(W0),
        f"content x edges (beta={beta:g})": part,
    }
    for name, p in cands.items():
        c, npairs, rate = coherence(p, pairs, y_te, repo_of)
        sz = collections.Counter(p)
        cov = sum(v for k_, v in sz.items() if v > 1 and k_ != -1) / n
        print(f"{name:<34}{len(set(p)):>7}{cov:>9.0%}{c:>11.2f}x")

    print(f"\n  the bar is directory structure. Louvain chose {len(sizes)} parts covering "
          f"{covered:.0%} of files.")
    print(f"  largest part {max(sizes.values())} files, singletons {sum(1 for v in sizes.values() if v == 1)}")

    np.save("partition.npy", part)
    print("\n  partition saved to partition.npy")


if __name__ == "__main__":
    main()
