"""S2 — score every predictor against git co-change, the one label nobody chose.

Everything measured so far has been internal: the geometry validated against nulls
built from the same geometry. That can establish that a pattern is not noise, but
never that it is USEFUL. Git history is external. Two files edited in the same
commit are coupled in a way no one selected to flatter this programme, and there
are 3153 commits of it across the two repositories.

The question: does anything derived from the typed graph predict co-change better
than the obvious alternatives?

Predictors compared
  random                 sanity floor
  same directory         the baseline everyone forgets to beat, and usually cannot
  same subsystem         the INCUMBENT 11-way partition already in the graph
  embedding cosine       raw 4096-d FastRP similarity, no graph at all
  graph adjacency        a typed edge exists, deduplicated
  graph 2-hop            adjacency plus its square, i.e. shared neighbours
  affinity               per-signature weights from the identity-model error of 6d
  affinity diffused      the above, propagated two hops

Method notes that decide whether the number means anything

  * WITHIN-REPO ONLY. The two repositories have separate histories, so a
    cross-repo pair can never co-change. Including such pairs would hand a free
    win to any predictor that implicitly encodes which repo a file is in --
    which every one of these does. This is the single easiest way to get a
    beautiful and meaningless AUC here.

  * Commits touching more than 30 files are dropped. A mass rename produces
    hundreds of files and tens of thousands of spurious pairs, and co-change
    counts are otherwise dominated by a handful of refactors.

  * Each surviving commit contributes total weight 1, split over its pairs, so a
    12-file commit does not outvote twelve 2-file commits.

  * Only files that actually appear in the history are scored. A file the graph
    knows about but git never mentions under that path (renamed at some point,
    since --name-only reports historical paths) carries no label and is excluded
    rather than counted as a negative.
"""
import collections
import itertools
import subprocess

import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS = "CheckItOutV3"
ROOT = "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
MAX_COMMIT_FILES = 30

# identity-model held-out error per signature, from 6d. Below 1.0 the endpoints
# are closer than the mean pair; at or above 1.0 they are not, and the signature
# earns no weight rather than a negative one.
IDENTITY_ERR = {
    ("Actor", "PERFORMS", "Process"): 0.43,
    ("Actor", "INJECTS", "Process"): 0.43,
    ("Actor", "INJECTS", "Resource"): 0.66,
    ("Actor", "ACCESSES", "Resource"): 0.66,
    ("Process", "MODIFIES", "Resource"): 0.64,
    ("Process", "USES", "Resource"): 0.81,
    ("Process", "INJECTS", "Resource"): 0.81,
    ("Process", "INJECTS", "Process"): 0.82,
    ("Process", "CALLS", "Process"): 0.82,
    ("Actor", "IMPORTS", "Process"): 0.82,
    ("Process", "IMPORTS", "Process"): 1.02,
    ("Rule", "IMPORTS", "Resource"): 1.01,
    ("Rule", "IMPORTS", "Process"): 1.04,
    ("Resource", "IMPORTS", "Resource"): 1.26,
    ("Process", "IMPORTS", "Resource"): 1.23,
    ("Actor", "IMPORTS", "Resource"): 1.52,
    ("Resource", "INJECTS", "Resource"): 1.21,
    ("Resource", "ALGEBRA_VIOLATION", "Resource"): 1.21,
    ("Resource", "EXTENDS", "Resource"): 1.47,
    ("Actor", "EXTENDS", "Resource"): 2.45,
    ("Process", "EXTENDS", "Resource"): 2.04,
    ("Resource", "IMPLEMENTS", "Resource"): 1.21,
}
DEFAULT_ERR = 1.0


# ------------------------------------------------------------------ co-change

def cochange(repo):
    """{(a,b): weight} over repo-relative paths, plus the set of files seen."""
    out = subprocess.run(
        ["git", "-C", ROOT + repo, "log", "--all", "--pretty=format:\x01%H", "--name-only"],
        capture_output=True, text=True, errors="replace").stdout

    pairs, seen, kept, dropped = collections.Counter(), set(), 0, 0
    for block in out.split("\x01")[1:]:
        lines = [l.strip() for l in block.splitlines()[1:] if l.strip()]
        files = sorted(set(lines))
        if not files:
            continue
        seen.update(files)
        if len(files) > MAX_COMMIT_FILES:
            dropped += 1
            continue
        kept += 1
        if len(files) < 2:
            continue
        w = 2.0 / (len(files) * (len(files) - 1))
        for a, b in itertools.combinations(files, 2):
            pairs[(a, b)] += w
    return pairs, seen, kept, dropped


# ------------------------------------------------------------------ predictors

def auc(scores, labels):
    """Rank-based AUC, ties averaged. 0.5 = no better than a coin."""
    order = np.argsort(scores, kind="mergesort")
    s, y = np.asarray(scores)[order], np.asarray(labels)[order]
    ranks = np.empty(len(s), dtype=np.float64)
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


def precision_at_k(scores, labels, k):
    idx = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    return float(np.asarray(labels)[idx].mean())


def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with driver.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
                "RETURN id(n) AS id, n.file_path AS fp, n.entity_type AS et, "
                "n.subsystem_id AS sub, n.embedding AS emb", ns=NS))
            edges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "RETURN id(a) AS s, id(b) AS t, type(r) AS k, "
                "a.entity_type AS st, b.entity_type AS tt", ns=NS))
    finally:
        driver.close()

    idx, repo_of, rel_of = {}, {}, {}
    for i, r in enumerate(nodes):
        idx[r["id"]] = i
        fp = (r["fp"] or "").replace("\\", "/")
        rest = fp[len(ROOT):] if fp.startswith(ROOT) else fp
        parts = rest.split("/", 1)
        repo_of[i] = parts[0] if len(parts) == 2 else "?"
        rel_of[i] = parts[1] if len(parts) == 2 else rest
    n = len(nodes)
    X = np.array([r["emb"] for r in nodes], dtype=np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    sub = np.array([r["sub"] if r["sub"] is not None else -1 for r in nodes])
    dirs = np.array([rel_of[i].rsplit("/", 1)[0] if "/" in rel_of[i] else "" for i in range(n)])

    print(f"graph: {n} nodes across {len(set(repo_of.values()))} repos, "
          f"{len(edges)} typed edges, {len(set(sub))} incumbent subsystems\n")

    # ---- labels
    label = collections.Counter()
    coverage = {}
    for repo in REPOS:
        pairs, seen, kept, dropped = cochange(repo)
        here = {rel_of[i]: i for i in range(n) if repo_of[i] == repo}
        hit = sum(1 for p in here if p in seen)
        coverage[repo] = (len(here), hit, kept, dropped)
        for (a, b), w in pairs.items():
            ia, ib = here.get(a), here.get(b)
            if ia is not None and ib is not None:
                label[(min(ia, ib), max(ia, ib))] += w
        print(f"{repo}: {kept} commits kept, {dropped} dropped as >{MAX_COMMIT_FILES} files; "
              f"{hit}/{len(here)} graph files found in history")

    # only score files git actually knows under their current path
    scored_nodes = set()
    for repo in REPOS:
        _, seen, _, _ = cochange(repo)
        scored_nodes |= {i for i in range(n) if repo_of[i] == repo and rel_of[i] in seen}
    scored = sorted(scored_nodes)
    print(f"\nscoring {len(scored)} files that appear in history under their current path")

    # ---- graph structures, deduplicated
    adj = np.zeros((n, n))
    aff = np.zeros((n, n))
    seen_pair_sig = set()
    for e in edges:
        u, v = idx.get(e["s"]), idx.get(e["t"])
        if u is None or v is None or u == v:
            continue
        adj[u, v] = adj[v, u] = 1.0
        key = (min(u, v), max(u, v), e["st"], e["k"], e["tt"])
        if key in seen_pair_sig:            # F18: 95.6% of INJECTS duplicates another edge
            continue
        seen_pair_sig.add(key)
        err = IDENTITY_ERR.get((e["st"], e["k"], e["tt"]), DEFAULT_ERR)
        w = max(0.0, 1.0 - err)
        if w > 0:
            aff[u, v] = aff[v, u] = max(aff[u, v], w)

    deg = np.maximum(adj.sum(axis=1, keepdims=True), 1.0)
    adj2 = adj + 0.5 * (adj / deg) @ adj
    daff = aff + 0.5 * (aff / np.maximum(aff.sum(axis=1, keepdims=True), 1e-9)) @ aff

    # ---- build the evaluation set, WITHIN REPO ONLY
    rng = np.random.default_rng(0)
    rows, ys = [], []
    for a, b in itertools.combinations(scored, 2):
        if repo_of[a] != repo_of[b]:
            continue                          # separate histories: can never co-change
        rows.append((a, b))
        ys.append(1 if label.get((a, b), 0.0) > 0 else 0)
    ys = np.array(ys)
    A = np.array([r[0] for r in rows])
    B = np.array([r[1] for r in rows])
    print(f"evaluation set: {len(rows):,} within-repo pairs, "
          f"{ys.sum():,} positives ({ys.mean():.2%} base rate)\n")

    preds = {
        "random": rng.random(len(rows)),
        "same directory": (dirs[A] == dirs[B]).astype(float),
        "same subsystem (incumbent)": (sub[A] == sub[B]).astype(float),
        "embedding cosine": np.einsum("ij,ij->i", X[A], X[B]),
        "graph adjacency": adj[A, B],
        "graph 2-hop": adj2[A, B],
        "affinity (6d weights)": aff[A, B],
        "affinity diffused": daff[A, B],
    }

    k = max(100, int(ys.sum()))
    print(f"{'predictor':<30}{'AUC':>8}{'P@' + str(k):>10}   lift over base rate")
    for name, sc in preds.items():
        a_, p_ = auc(sc, ys), precision_at_k(sc, ys, k)
        print(f"{name:<30}{a_:>8.3f}{p_:>10.3f}   {p_ / max(ys.mean(), 1e-12):>6.1f}x")

    print("\n  AUC 0.5 is a coin flip. The bar that matters is `same directory` --")
    print("  if the geometry cannot beat knowing which folder a file is in, it is")
    print("  not earning its complexity.")

    np.savez("cochange_eval.npz", A=A, B=B, y=ys,
             **{k_.replace(" ", "_"): v for k_, v in preds.items()})
    print("\n  saved to cochange_eval.npz")


if __name__ == "__main__":
    main()
