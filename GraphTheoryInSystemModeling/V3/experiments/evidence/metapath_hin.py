"""V-arc — the V2 schema is a Heterogeneous Information Network. Use it as one.

The thing V2 built and V3 never exploited: 6 typed node types (Actor, Process,
Resource, Rule, Context, Event) crossed with 18 typed edge types is precisely a
Heterogeneous Information Network, and HINs have a mature literature this
programme has not been using — meta-paths, PathSim, weighted meta-path
combination, learned meta-path attention.

Why this is the right instrument for the problem that has resisted everything
else. §6n showed a subsystem is a VERTICAL SLICE — a controller, a service, a
DTO — and that no pairwise similarity can see it, because those three files do not
resemble one another. A META-PATH can:

    Actor -PERFORMS-> Process -USES-> Resource

connects a controller to a DTO THROUGH the service that binds them. The
connection is not similarity, it is a typed walk. That is the missing primitive,
and it is exactly the schema V2 specified.

PathSim (Sun et al.) for a symmetric meta-path P:

    s_P(x,y) = 2 · |paths x→y via P| / (|paths x→x via P| + |paths y→y via P|)

computed from the commuting matrix M_P = A_1 A_2 … A_k, with A_i the typed
adjacency for each hop. Asymmetric paths get cosine on the commuting matrix rows
instead, since PathSim's normalisation assumes symmetry.

Multiple meta-paths are combined with weights λ_i ≥ 0, Σλ = 1 — the "changed
weights" idea, and the literature's standard answer to the fact that no single
meta-path captures a complex HIN.

Three questions, all against held-out co-change over 20 splits:

  Q1  Does any single meta-path beat the untyped graph baseline (2-hop, 0.6211)?
      If typed walks are no better than untyped ones, the schema is decoration.
  Q2  Does a weighted combination beat the best single meta-path?
  Q3  THE ONE THAT MATTERS: does adding meta-path similarity to the current best
      (content + lexical, 0.8411) improve it? Gate G-a as corrected: it earns its
      place only by beating the better existing lens in ≥18 of 20 splits.
"""
import collections
import itertools
import subprocess

import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
N_SPLITS, SEED = 20, 42

# Meta-paths worth trying, written as (source type, [(relation, target type), ...]).
# Chosen from the signatures that passed the reliability gate (F11/F12) plus the
# two structural relations that carry the most edges, rather than by enumeration.
METAPATHS = [
    ("A-P-A", "Actor", [("PERFORMS", "Process")]),
    ("A-R-A", "Actor", [("ACCESSES", "Resource")]),
    ("P-R-P", "Process", [("USES", "Resource")]),
    ("P-R-P/mod", "Process", [("MODIFIES", "Resource")]),
    ("A-P-R", "Actor", [("PERFORMS", "Process"), ("USES", "Resource")]),
    ("A-P-R/mod", "Actor", [("PERFORMS", "Process"), ("MODIFIES", "Resource")]),
    ("X-imp-Y", None, [("IMPORTS", None)]),
    ("X-inj-Y", None, [("INJECTS", None)]),
    ("X-imp-imp", None, [("IMPORTS", None), ("IMPORTS", None)]),
]


def auc(scores, labels):
    order = np.argsort(scores, kind="mergesort")
    s, y = np.asarray(scores, float)[order], np.asarray(labels)[order]
    ranks = np.empty(len(s))
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


def main():
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
                "RETURN id(n) AS id, n.file_path AS fp, n.entity_type AS et, "
                "n.embedding AS emb", ns=NS))
            edges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "RETURN id(a) AS s, id(b) AS t, type(r) AS k", ns=NS))
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

    # typed adjacency per relation, deduplicated
    A = collections.defaultdict(lambda: np.zeros((n, n)))
    seen = set()
    for e in edges:
        u, v = idx.get(e["s"]), idx.get(e["t"])
        if u is None or v is None or u == v:
            continue
        key = (u, v, e["k"])
        if key in seen:
            continue
        seen.add(key)
        A[e["k"]][u, v] = 1.0
    print(f"{n} nodes, {len(A)} relation types, "
          f"{sum(int(m.sum()) for m in A.values())} deduplicated typed edges\n")

    def commuting(src_type, hops):
        """M = A_1 A_2 … A_k with node-type masks applied at each stage."""
        M = np.eye(n)
        if src_type:
            M = M * (et == src_type)[:, None]
        for rel, tgt in hops:
            if rel not in A:
                return None
            step = A[rel]
            if tgt:
                step = step * (et == tgt)[None, :]
            M = M @ step
            if not M.any():
                return None
        return M

    def pathsim(M):
        """Symmetric PathSim over the commuting matrix; falls back to row cosine."""
        S = M @ M.T                       # x→…→z←…←y : shared endpoints
        d = np.diag(S).copy()
        denom = d[:, None] + d[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(denom > 0, 2.0 * S / np.maximum(denom, 1e-12), 0.0)
        return out

    sims = {}
    for name, src, hops in METAPATHS:
        M = commuting(src, hops)
        if M is None:
            print(f"  meta-path {name}: no instances, skipped")
            continue
        S = pathsim(M)
        S = np.maximum(S, S.T)
        nz = int((np.triu(S, 1) > 0).sum())
        if nz < 50:
            print(f"  meta-path {name}: only {nz} connected pairs, skipped")
            continue
        sims[name] = S
        print(f"  meta-path {name:<12} {nz:>7,} connected pairs")

    # labels
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

    # existing best: content + lexical rank mix (F75)
    import math
    import re
    import scipy.sparse as sp
    _TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

    def ident(text):
        for w in _TOK.findall(text):
            w = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w)
            for p_ in re.split(r"[_\s]+", w):
                if len(p_) > 2:
                    yield p_.lower()

    bags = []
    for r in nodes:
        try:
            txt = open(r["fp"], encoding="utf-8", errors="replace").read()[:24_000]
        except Exception:
            txt = ""
        bags.append(collections.Counter(ident(txt)))
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
    T = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)))
    nr = np.sqrt(T.multiply(T).sum(axis=1)).A.ravel()
    nr[nr == 0] = 1.0
    Td = np.asarray((sp.diags(1 / nr) @ T).todense())

    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[o] = np.arange(len(v))
        return r / max(len(v) - 1, 1)

    q = np.einsum("ij,ij->i", X[P[:, 0]], X[P[:, 1]])
    t = np.einsum("ij,ij->i", Td[P[:, 0]], Td[P[:, 1]])
    base_mix = 0.5 * rank(q) + 0.5 * rank(t)
    mp_vecs = {k: v[P[:, 0], P[:, 1]] for k, v in sims.items()}
    uniform = np.mean([rank(v) for v in mp_vecs.values()], axis=0) if mp_vecs else None

    res = collections.defaultdict(list)
    for s_ in range(N_SPLITS):
        rng = np.random.default_rng(SEED + s_)
        lab = collections.Counter()
        for rp in REPOS:
            keep = rng.random(len(raw[rp])) < 0.5
            for ids, k in zip(raw[rp], keep):
                if k:
                    continue
                for a, b in itertools.combinations(ids, 2):
                    lab[(a, b)] += 1
        y = np.array([1 if lab.get((a, b), 0) else 0 for a, b in pairs])
        for k, v in mp_vecs.items():
            res[f"metapath {k}"].append(auc(v, y))
        if uniform is not None:
            res["metapaths, uniform weights"].append(auc(uniform, y))
        res["content+lexical (current best)"].append(auc(base_mix, y))
        if uniform is not None:
            for w in (0.1, 0.2, 0.3):
                res[f"best + metapaths w={w}"].append(
                    auc((1 - w) * base_mix + w * uniform, y))

    print("\n\nQ1/Q2 — META-PATHS ALONE, vs the untyped 2-hop baseline of 0.6211\n")
    print(f"{'signal':<34}{'AUC':>9}{'sd':>8}")
    for k in sorted(res):
        if k.startswith("metapath"):
            v = np.array(res[k])
            print(f"{k:<34}{v.mean():>9.4f}{v.std():>8.4f}")

    print("\n\nQ3 — DOES THE TYPED SCHEMA ADD TO THE CURRENT BEST?\n")
    best = np.array(res["content+lexical (current best)"])
    print(f"{'combination':<34}{'AUC':>9}{'sd':>8}{'vs best':>10}{'wins':>8}")
    print(f"{'content+lexical (current best)':<34}{best.mean():>9.4f}{best.std():>8.4f}"
          f"{0.0:>+10.4f}{'—':>8}")
    for k in sorted(res):
        if k.startswith("best +"):
            v = np.array(res[k])
            d = v - best
            print(f"{k:<34}{v.mean():>9.4f}{v.std():>8.4f}{d.mean():>+10.4f}"
                  f"{int((d > 0).sum()):>6}/{N_SPLITS}")
    print("\n  Gate G-a (corrected): a lens earns its place only by beating the better")
    print("  existing lens in >=18 of 20 splits. Anything less is decoration.")


if __name__ == "__main__":
    main()
