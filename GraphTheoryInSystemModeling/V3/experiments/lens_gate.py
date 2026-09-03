"""V3/V4 — the gate the triple-lens rebuild has to pass, and the baseline it must beat.

Written before the lenses exist so the acceptance criterion cannot be chosen after
seeing the numbers. Run it the moment HypatiaV4's shards finish.

GATE G-a — lens usefulness, NOT lens independence.
    V2 specifies lenses "orthogonal by design, correlation < 0.3" and the rebuild
    plan originally adopted that. It is the wrong gate, and measurement said so:
    Qwen3 similarity and TF-IDF similarity correlate at +0.643 — far above the
    threshold — and still combine for +0.0160 AUC in 20/20 splits. A correlation
    cutoff would have rejected a combination that demonstrably works.
    So the gate is the OUTCOME: a lens earns its place if adding it beats the
    BETTER of the existing lenses, paired across 20 commit splits, in >= 18.

GATE G-b — the bar is not the old single embedding.
    The current best is content + lexical rank-mix at 0.8411 +- 0.0082, not the
    0.8212 of the generic embedding alone. Beating the thing we already replaced
    is not progress.

Also reported, because a null result here is informative and must not be quietly
reframed: the pairwise correlation matrix between lenses, and each lens alone.
If S, B and T come back mutually correlated above ~0.95 the sockets are not
differentiating and the fault is the socket TEXT, not the model — instruction
conditioning was measured at only 5.5% separation, so it cannot rescue them.
"""
import collections
import itertools
import math
import re
import subprocess

import numpy as np
import scipy.sparse as sp
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
N_SPLITS, SEED = 20, 42
LENSES = [("S semantic", "semantic_embedding"),
          ("B behavioural", "behavioral_embedding"),
          ("T structural", "structural_embedding")]

_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def idents(text):
    for w in _TOK.findall(text):
        w = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w)
        for p in re.split(r"[_\s]+", w):
            if len(p) > 2:
                yield p.lower()


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


def rank(v):
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[o] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def main():
    cols = ", ".join(f"n.`{p}` AS {k}" for k, (_, p) in
                     zip("abc", LENSES))
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                f"MATCH (n:EntityDetail {{namespace:$ns}}) WHERE n.embedding IS NOT NULL "
                f"RETURN id(n) AS id, n.file_path AS fp, n.embedding AS generic, "
                f"n.lens_status AS st, {cols}", ns=NS))
    finally:
        drv.close()

    n = len(nodes)
    have = {k: sum(1 for r in nodes if r[k]) for k in "abc"}
    print(f"{n} nodes; lenses present: " +
          ", ".join(f"{nm} {have[k]}" for k, (nm, _) in zip("abc", LENSES)))
    ready = [k for k in "abc" if have[k] >= 0.95 * n]
    if not ready:
        print("\n  No lens is populated yet. Run this after the HypatiaV4 shards finish.")
        print("  (Written now so the acceptance criterion is fixed before the numbers.)")
        return

    repo_of, rel_of = {}, {}
    for i, r in enumerate(nodes):
        rest = (r["fp"] or "").replace("\\", "/")
        rest = rest[len(ROOT):] if rest.startswith(ROOT) else rest
        p = rest.split("/", 1)
        repo_of[i], rel_of[i] = (p[0], p[1]) if len(p) == 2 else ("?", rest)

    def unit(M):
        return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)

    G = unit(np.array([r["generic"] for r in nodes], dtype=np.float64))
    L = {}
    for k, (nm, _) in zip("abc", LENSES):
        if k in ready:
            L[nm] = unit(np.array([r[k] for r in nodes], dtype=np.float64))

    bags = []
    for r in nodes:
        try:
            txt = open(r["fp"], encoding="utf-8", errors="replace").read()[:24_000]
        except Exception:
            txt = ""
        bags.append(collections.Counter(idents(txt)))
    df = collections.Counter()
    for b in bags:
        df.update(b.keys())
    vocab = {w: i for i, (w, _) in enumerate(df.most_common(20_000))}
    rows, cs, vs = [], [], []
    for i, b in enumerate(bags):
        for w, c in b.items():
            j = vocab.get(w)
            if j is not None:
                rows.append(i); cs.append(j)
                vs.append((1 + math.log(c)) * math.log(n / df[w]))
    T = sp.csr_matrix((vs, (rows, cs)), shape=(n, len(vocab)))
    nr = np.sqrt(T.multiply(T).sum(axis=1)).A.ravel()
    nr[nr == 0] = 1.0
    Td = np.asarray((sp.diags(1 / nr) @ T).todense())

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

    sims = {"generic embedding": np.einsum("ij,ij->i", G[P[:, 0]], G[P[:, 1]]),
            "lexical TF-IDF": np.einsum("ij,ij->i", Td[P[:, 0]], Td[P[:, 1]])}
    for nm, M in L.items():
        sims[nm] = np.einsum("ij,ij->i", M[P[:, 0]], M[P[:, 1]])

    print("\nPAIRWISE CORRELATION BETWEEN SIGNALS\n")
    names = list(sims)
    print("            " + "".join(f"{x[:11]:>13}" for x in names))
    for a in names:
        print(f"{a[:11]:<12}" + "".join(
            f"{np.corrcoef(sims[a], sims[b])[0,1]:>13.3f}" for b in names))
    print("\n  reported for information only — correlation is NOT the gate (F76)")

    baseline = 0.5 * rank(sims["generic embedding"]) + 0.5 * rank(sims["lexical TF-IDF"])
    cands = {"BASELINE content+lexical": baseline}
    for nm in L:
        cands[f"baseline + {nm}"] = (2 * baseline + rank(sims[nm])) / 3.0
    if len(L) == 3:
        allmix = np.mean([rank(sims[nm]) for nm in L], axis=0)
        cands["three lenses only"] = allmix
        cands["baseline + all three"] = 0.5 * baseline + 0.5 * allmix

    res = collections.defaultdict(list)
    for t in range(N_SPLITS):
        rng = np.random.default_rng(SEED + t)
        lab = collections.Counter()
        for rp in REPOS:
            keep = rng.random(len(raw[rp])) < 0.5
            for ids, k in zip(raw[rp], keep):
                if k:
                    continue
                for a, b in itertools.combinations(ids, 2):
                    lab[(a, b)] += 1
        y = np.array([1 if lab.get((a, b), 0) else 0 for a, b in pairs])
        for nm, v in sims.items():
            res[f"alone: {nm}"].append(auc(v, y))
        for nm, v in cands.items():
            res[nm].append(auc(v, y))

    print("\nEACH SIGNAL ALONE\n")
    print(f"{'signal':<34}{'AUC':>9}{'sd':>8}")
    for k in [f"alone: {x}" for x in sims]:
        v = np.array(res[k])
        print(f"{k:<34}{v.mean():>9.4f}{v.std():>8.4f}")

    print("\nGATE — does a lens beat the better existing lens in >=18/20 splits?\n")
    base = np.array(res["BASELINE content+lexical"])
    print(f"{'combination':<34}{'AUC':>9}{'sd':>8}{'vs base':>10}{'wins':>8}")
    print(f"{'BASELINE content+lexical':<34}{base.mean():>9.4f}{base.std():>8.4f}"
          f"{0.0:>+10.4f}{'—':>8}")
    passed = []
    for k in cands:
        if k.startswith("BASELINE"):
            continue
        v = np.array(res[k])
        d = v - base
        w = int((d > 0).sum())
        if w >= 18:
            passed.append(k)
        print(f"{k:<34}{v.mean():>9.4f}{v.std():>8.4f}{d.mean():>+10.4f}"
              f"{w:>6}/{N_SPLITS}")
    if passed:
        print(f"\n  PASSED: {', '.join(passed)}")
    else:
        print("\n  PASSED: nothing — the triple lens does not beat content+lexical.")
        print("  That is a committable negative result, not a reason to retune.")


if __name__ == "__main__":
    main()
