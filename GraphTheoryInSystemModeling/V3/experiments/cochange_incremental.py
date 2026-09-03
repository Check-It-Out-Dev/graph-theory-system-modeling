"""S2b — two fairness checks, and the question that actually matters.

The first pass reported AUC 0.843 for raw content-embedding cosine and 0.511 for
the affinity built from tonight's typed-relation work. Before accepting either,
two corrections:

  1. Sparse predictors were scored unfairly. Adjacency and affinity are nonzero on
     a few thousand of 565,635 pairs, so precision@12438 pads them with ten
     thousand arbitrary ties, and AUC over a mostly-tied vector sits near 0.5 by
     construction. The fair questions are how precise a sparse predictor is WHERE
     IT FIRES, and how it ranks within its own support.

  2. "Does the graph beat the embedding" is the wrong question. The embedding is
     Qwen3-Embedding-8B over file CONTENT -- it is reading the code. The graph was
     built to add structure that reading cannot see. So the question is whether
     the graph adds anything ON TOP: does embedding + graph beat embedding alone,
     and among pairs the embedding considers equally similar, does an edge still
     carry information?

The second is a stratified test and it is the honest one. If two files look
equally alike to a model that has read them both, and one pair is joined by a
typed edge while the other is not, does the edge predict co-change? That isolates
the graph's contribution from the content's.
"""
import numpy as np

SEED = 0


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
    d = np.load("cochange_eval.npz")
    y = d["y"]
    base = y.mean()
    emb = d["embedding_cosine"]
    print(f"{len(y):,} within-repo pairs, {y.sum():,} positives, base rate {base:.2%}\n")

    # ---- 1. sparse predictors judged where they actually fire
    print("1. SPARSE PREDICTORS, JUDGED WHERE THEY FIRE\n")
    print(f"{'predictor':<28}{'fires on':>10}{'precision':>11}{'lift':>8}{'recall':>9}")
    for name in ["graph_adjacency", "graph_2-hop", "affinity_(6d_weights)",
                 "affinity_diffused", "same_directory", "same_subsystem_(incumbent)"]:
        s = d[name]
        m = s > 0
        if m.sum() == 0:
            continue
        prec = y[m].mean()
        print(f"{name.replace('_', ' '):<28}{m.sum():>10,}{prec:>11.3f}"
              f"{prec / base:>7.1f}x{y[m].sum() / y.sum():>9.1%}")
    print("\n  precision here is the chance a pair it points at really did co-change;")
    print("  recall is how much of the total coupling it manages to point at at all")

    # ---- 2. does the graph add anything on top of content?
    print("\n\n2. DOES THE GRAPH ADD ANYTHING ON TOP OF READING THE CODE?\n")
    print(f"  embedding alone                       AUC {auc(emb, y):.4f}")
    for name in ["graph_adjacency", "graph_2-hop", "affinity_diffused"]:
        g = d[name].astype(float)
        g = g / max(g.max(), 1e-12)
        best = max(((auc(emb + lam * g, y), lam) for lam in
                    (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)), key=lambda t: t[0])
        delta = best[0] - auc(emb, y)
        flag = "adds" if delta > 0.002 else ("neutral" if delta > -0.002 else "HURTS")
        print(f"  embedding + {name.replace('_', ' '):<24} AUC {best[0]:.4f} "
              f"(lam {best[1]:g})  {delta:+.4f}  {flag}")

    # ---- 3. the stratified test: equal content similarity, edge or no edge
    print("\n\n3. STRATIFIED — AMONG PAIRS THE CONTENT MODEL FINDS EQUALLY SIMILAR,\n"
          "   DOES A TYPED EDGE STILL PREDICT CO-CHANGE?\n")
    qs = np.quantile(emb, np.linspace(0, 1, 11))
    adj = d["graph_adjacency"] > 0
    print(f"{'content-similarity decile':<28}{'pairs':>9}{'no edge':>10}{'edge':>10}"
          f"{'n(edge)':>9}{'ratio':>8}")
    tot_e, tot_n = [], []
    for i in range(10):
        lo, hi = qs[i], qs[i + 1]
        m = (emb >= lo) & (emb < hi if i < 9 else emb <= hi)
        if m.sum() < 50:
            continue
        e, ne = m & adj, m & ~adj
        if e.sum() < 5:
            continue
        pe, pn = y[e].mean(), y[ne].mean()
        tot_e.append((e.sum(), pe))
        tot_n.append((ne.sum(), pn))
        print(f"  decile {i + 1} [{lo:+.2f},{hi:+.2f}]{'':<6}{m.sum():>9,}"
              f"{pn:>10.3f}{pe:>10.3f}{e.sum():>9,}"
              f"{(pe / pn if pn > 0 else np.nan):>8.1f}x")
    if tot_e:
        we = sum(c * p for c, p in tot_e) / sum(c for c, _ in tot_e)
        wn = sum(c * p for c, p in tot_n) / sum(c for c, _ in tot_n)
        print(f"\n  pooled across deciles: edge {we:.3f} vs no edge {wn:.3f} "
              f"-> {we / max(wn, 1e-12):.1f}x")
        print("  this is the graph's contribution with content held fixed. A ratio near")
        print("  1.0 would mean the typed edges are redundant given the file contents.")

    # ---- 4. is the incumbent partition really worse than random?
    print("\n\n4. THE INCUMBENT 11-SUBSYSTEM PARTITION\n")
    same = d["same_subsystem_(incumbent)"] > 0
    print(f"  pairs in the same subsystem : {same.sum():,} ({same.mean():.1%})")
    print(f"  co-change rate, same subsystem   : {y[same].mean():.4f}")
    print(f"  co-change rate, different       : {y[~same].mean():.4f}")
    print(f"  ratio                            : {y[same].mean() / max(y[~same].mean(), 1e-12):.2f}x")
    print("\n  a useful partition puts coupled files together, so this ratio must")
    print("  exceed 1. Below 1 means the partition is anti-correlated with real")
    print("  coupling -- worse than assigning subsystems at random.")

    # directory baseline for comparison, same form
    sd = d["same_directory"] > 0
    print(f"\n  for comparison, same directory   : {y[sd].mean():.4f} vs "
          f"{y[~sd].mean():.4f}  -> {y[sd].mean() / max(y[~sd].mean(), 1e-12):.2f}x")


if __name__ == "__main__":
    main()
