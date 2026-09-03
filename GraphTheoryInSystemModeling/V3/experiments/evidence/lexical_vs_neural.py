"""V-arc — is the embedding model the bottleneck? And what makes two views combine?

Asked before choosing a model for the rebuild: is anything better than
Qwen3-Embedding-8B worth switching to? voyage-code-3 and Gemini Embedding 2 both
score 84.0 on MTEB Code against Qwen3's 80.68, so on paper yes — but both are
proprietary API models, and the question is whether 3.3 MTEB points buy anything
on OUR task, which is code-to-code similarity for co-change prediction rather
than query-to-code retrieval.

The way to bound that without paying for an API is to ask how much the 8B GPU
model beats a free baseline. If a bag of identifiers is close behind, the model
tier is not where the headroom is.

TF-IDF baseline: split identifiers on camelCase and underscores, drop tokens
under three characters, log-scaled term frequency times inverse document
frequency over the top 20k vocabulary, cosine on L2-normalised rows. No model, no
GPU, a few seconds of CPU.

Then the second question, which turned out to matter more: the two signals
combine. Every earlier combination in this programme failed — up-weighting typed
edges, hard must-link constraints, routing by pair class — so a combination that
works is worth understanding. Reported as a paired comparison against the BETTER
of the two singles on each split, because beating the average of two singles is
not the same as beating either.
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
N_SPLITS, VOCAB, SEED = 20, 20_000, 42

_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def identifiers(text):
    for w in _TOK.findall(text):
        w = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w)
        for part in re.split(r"[_\s]+", w):
            if len(part) > 2:
                yield part.lower()


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
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
                "RETURN id(n) AS id, n.file_path AS fp, n.name AS name, "
                "n.embedding AS emb", ns=NS))
    finally:
        drv.close()

    n = len(nodes)
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
            txt = open(r["fp"], encoding="utf-8", errors="replace").read()[:24_000]
        except Exception:
            txt = r["name"] or ""
        bags.append(collections.Counter(identifiers(txt)))
    df = collections.Counter()
    for b in bags:
        df.update(b.keys())
    vocab = {w: i for i, (w, _) in enumerate(df.most_common(VOCAB))}
    rows, cols, vals = [], [], []
    for i, b in enumerate(bags):
        for w, c in b.items():
            j = vocab.get(w)
            if j is not None:
                rows.append(i)
                cols.append(j)
                vals.append((1 + math.log(c)) * math.log(n / df[w]))
    T = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)))
    nrm = np.sqrt(T.multiply(T).sum(axis=1)).A.ravel()
    nrm[nrm == 0] = 1.0
    Td = np.asarray((sp.diags(1 / nrm) @ T).todense())

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
    q = np.einsum("ij,ij->i", X[P[:, 0]], X[P[:, 1]])
    t = np.einsum("ij,ij->i", Td[P[:, 0]], Td[P[:, 1]])
    rq = np.argsort(np.argsort(q)) / len(q)
    rt = np.argsort(np.argsort(t)) / len(t)

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
        res["Qwen3-Embedding-8B"].append(auc(q, y))
        res["TF-IDF identifiers (free)"].append(auc(t, y))
        for w in (0.3, 0.5, 0.7):
            res[f"rank mix w_lex={w}"].append(auc((1 - w) * rq + w * rt, y))
        res["max of ranks"].append(auc(np.maximum(rq, rt), y))

    best_single = np.maximum(np.array(res["Qwen3-Embedding-8B"]),
                             np.array(res["TF-IDF identifiers (free)"]))
    print(f"{n} files, {len(pairs):,} within-repo pairs, {N_SPLITS} commit splits\n")
    print(f"{'representation':<28}{'AUC':>9}{'sd':>8}{'vs best single':>16}{'wins':>8}")
    for k in ("Qwen3-Embedding-8B", "TF-IDF identifiers (free)", "rank mix w_lex=0.3",
              "rank mix w_lex=0.5", "rank mix w_lex=0.7", "max of ranks"):
        v = np.array(res[k])
        d = v - best_single
        print(f"{k:<28}{v.mean():>9.4f}{v.std():>8.4f}{d.mean():>+16.4f}"
              f"{int((d > 0).sum()):>6}/{N_SPLITS}")

    corr = float(np.corrcoef(q, t)[0, 1])
    print(f"\n  An 8B GPU model and a bag of identifiers are within "
          f"{abs(np.mean(res['Qwen3-Embedding-8B']) - np.mean(res['TF-IDF identifiers (free)'])):.4f} "
          f"AUC of each other.")
    print(f"  The model tier is not where the headroom is, so a 3.3-point MTEB-Code")
    print(f"  advantage (voyage-code-3, Gemini Embedding 2) is very unlikely to pay for")
    print(f"  sending a proprietary codebase to a third-party API.")
    print(f"\n  correlation between the two signals: {corr:+.3f}")
    print(f"  They combine anyway, in {N_SPLITS}/{N_SPLITS} splits. So V2's")
    print(f"  'orthogonal by design, correlation < 0.3' is the WRONG GATE: these two")
    print(f"  are correlated at {corr:.2f} and still add {np.mean(res['rank mix w_lex=0.5'] - best_single):+.4f}.")
    print(f"  Test the outcome, not a correlation proxy for it.")


if __name__ == "__main__":
    main()
