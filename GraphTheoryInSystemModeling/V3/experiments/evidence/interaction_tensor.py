"""
The relation-interaction tensor, and what it says the final topology should be.

The separate measurements — who shares which files, who reads which directions of
the original space, who pulls against whom, what happens going around a loop —
are all indexed by the same thing: pairs and triples of relations. So they are
not separate results. They are slices of one object.

  T2[i, j, :]   five numbers per ordered pair
                  0  support Jaccard        how much of the GRAPH they share
                  1  subspace overlap       how much of R^4096 they share,
                                            as cos of the mean principal angle
                  2  correction cosine      signed structure: together or opposed
                  3  orientation            +1 deformable to identity, -1 reversed
                  4  conditioning           1/cond, i.e. 1 = rigid, 0 = collapsing

  T3[i, j, k]   holonomy around the closed loop i -> j -> k -> i,
                as rotation angle in degrees

The point of assembling it is not tidiness. Two relations should be ONE lens when
they share the graph, share the subspace, pull the same way, preserve orientation
and transition rigidly. They must stay SEPARATE when they do not. That turns the
relation set from a design decision into a measurement, which is the clue to what
the final topology should contain.
"""
import itertools
import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS = "CheckItOutV3"
RELS = ["PERFORMS", "USES", "MODIFIES", "CALLS", "ACCESSES"]
FEATURES = ["jaccard", "subspace", "delta_cos", "orient", "rigidity"]
LAM = 0.1
MERGE_THRESHOLD = 0.55


def load(driver):
    props = ", ".join(f"n.`proj_{r}` AS {r}" for r in RELS)
    q = (f"MATCH (n:EntityDetail {{namespace: $ns}}) "
         f"WHERE n.embedding IS NOT NULL AND n.rho0 IS NOT NULL "
         f"RETURN n.embedding AS emb, n.rho0 AS rho0, {props}")
    with driver.session() as s:
        rows = list(s.run(q, ns=NS))
    X = np.array([r["emb"] for r in rows], dtype=np.float64)
    R0 = np.array([r["rho0"] for r in rows], dtype=np.float64)
    P, sup = {}, {}
    for rel in RELS:
        M = np.array([r[rel] if r[rel] is not None else [0.0] * 8 for r in rows], dtype=np.float64)
        P[rel], sup[rel] = M, np.abs(M).sum(axis=1) > 0
    return X, R0, P, sup


def fit_rho(X, Y, mask, lam=LAM):
    Xs, Ys = X[mask], Y[mask]
    Xc, Yc = Xs - Xs.mean(axis=0), Ys - Ys.mean(axis=0)
    return Xc.T @ np.linalg.solve(Xc @ Xc.T + lam * np.eye(len(Xc)), Yc)


def subspace_overlap(A, B):
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0)))          # cos of mean principal angle


def delta_cos(P, R0, sup, a, b):
    m = sup[a] & sup[b]
    if m.sum() < 5:
        return np.nan
    da, db = P[a][m] - R0[m], P[b][m] - R0[m]
    na, nb = np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1)
    ok = (na > 1e-12) & (nb > 1e-12)
    return float(np.mean(np.sum(da[ok] * db[ok], axis=1) / (na[ok] * nb[ok])))


def build(X, R0, P, sup):
    n = len(RELS)
    rho = {r: fit_rho(X, P[r], sup[r]) for r in RELS}
    T2 = np.full((n, n, len(FEATURES)), np.nan)
    M = {}
    for i, a in enumerate(RELS):
        for j, b in enumerate(RELS):
            if i == j:
                continue
            m = sup[a] & sup[b]
            union = (sup[a] | sup[b]).sum()
            T2[i, j, 0] = m.sum() / union if union else 0.0
            T2[i, j, 1] = subspace_overlap(rho[a], rho[b])
            T2[i, j, 2] = delta_cos(P, R0, sup, a, b)
            if m.sum() >= 12:
                t = np.linalg.lstsq(P[a][m], P[b][m], rcond=None)[0]
                M[(a, b)] = t
                T2[i, j, 3] = np.sign(np.linalg.det(t))
                T2[i, j, 4] = 1.0 / max(np.linalg.cond(t), 1.0)
    T3 = np.full((n, n, n), np.nan)
    for (i, a), (j, b), (k, c) in itertools.permutations(list(enumerate(RELS)), 3):
        keys = [(a, b), (b, c), (c, a)]
        if all(key in M for key in keys):
            H = M[keys[0]] @ M[keys[1]] @ M[keys[2]]
            T3[i, j, k] = float(np.degrees(np.mean(np.abs(np.angle(np.linalg.eigvals(H))))))
    return T2, T3


def show(T2, T3):
    print("T2 — pairwise slices\n")
    for f, name in enumerate(FEATURES):
        print(f"  [{name}]")
        print("            " + "".join(f"{r[:8]:>10}" for r in RELS))
        for i, a in enumerate(RELS):
            cells = "".join("       —  " if np.isnan(T2[i, j, f]) else f"{T2[i, j, f]:>10.2f}"
                            for j in range(len(RELS)))
            print(f"  {a:<10}{cells}")
        print()

    print("T3 — holonomy rotation in degrees, minimum over the loops through each pair\n")
    print("            " + "".join(f"{r[:8]:>10}" for r in RELS))
    for i, a in enumerate(RELS):
        row = []
        for j in range(len(RELS)):
            vals = T3[i, j, :][~np.isnan(T3[i, j, :])]
            row.append("       —  " if not len(vals) else f"{vals.min():>10.1f}")
        print(f"  {a:<10}" + "".join(row))


def merge_score(T2, i, j):
    """One number per pair: should these be the same lens?

    Symmetrised, because 'is this one lens' is not a directed question. Each term
    is on [0,1] and they are multiplied, not averaged, so a single disqualifying
    property (opposed corrections, reversed orientation, a collapsing transition)
    vetoes the merge rather than being averaged away.
    """
    def sym(f):
        v = [T2[i, j, f], T2[j, i, f]]
        v = [x for x in v if not np.isnan(x)]
        return np.mean(v) if v else np.nan

    jac, sub, dcos, orient, rigid = (sym(f) for f in range(5))
    if any(np.isnan(x) for x in (jac, sub, dcos)):
        return np.nan
    same_direction = max(0.0, dcos)                      # opposed => 0
    orientation_ok = 1.0 if (np.isnan(orient) or orient > 0) else 0.0
    rigidity = 0.0 if np.isnan(rigid) else min(1.0, rigid * 10)
    return float((jac ** 0.5) * sub * same_direction * orientation_ok * max(rigidity, 0.05) ** 0.25)


def recommend(T2):
    print("\n\nWHAT THE TENSOR SAYS THE RELATION SET SHOULD BE\n")
    print(f"{'pair':<24}{'jaccard':>9}{'subspace':>10}{'delta':>8}{'orient':>8}{'merge':>8}")
    scores = {}
    for i, j in itertools.combinations(range(len(RELS)), 2):
        s = merge_score(T2, i, j)
        scores[(i, j)] = s
        o = T2[i, j, 3]
        print(f"{RELS[i] + ' & ' + RELS[j]:<24}"
              f"{T2[i, j, 0]:>9.2f}{T2[i, j, 1]:>10.2f}{T2[i, j, 2]:>8.2f}"
              f"{'—' if np.isnan(o) else ('+' if o > 0 else '-'):>8}"
              f"{'—' if np.isnan(s) else f'{s:.3f}':>8}")

    merges = [(p, s) for p, s in scores.items() if not np.isnan(s) and s >= MERGE_THRESHOLD]
    print(f"\n  merge threshold {MERGE_THRESHOLD}")
    if merges:
        for (i, j), s in sorted(merges, key=lambda t: -t[1]):
            print(f"    MERGE  {RELS[i]} + {RELS[j]}   (score {s:.3f})")
    else:
        print("    no pair qualifies — every relation is its own lens")
    kept = len(RELS) - len(merges)
    print(f"\n  {len(RELS)} measured relations -> {kept} distinct lenses")


def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        X, R0, P, sup = load(driver)
        T2, T3 = build(X, R0, P, sup)
        show(T2, T3)
        recommend(T2)
        np.savez("interaction_tensor.npz", T2=T2, T3=T3, rels=np.array(RELS))
        print("\n  tensor saved to interaction_tensor.npz")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
