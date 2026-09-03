"""
Does rho_k exist?

The V3 papers report a parameter budget of 102,553 weights for restriction maps
rho_k: R^4096 -> R^8, "independent of graph size". The pipeline never builds that
matrix — FastRP produces per-node coordinates, and the map exists only implicitly.

This script asks whether the matrix can be recovered, and whether it generalises:
fit rho_k by ridge regression from the node embedding to the per-relation FastRP
coordinates, and score it OUT OF SAMPLE with 5-fold cross-validation.

In-sample R^2 is meaningless here (n < d means an exact fit always exists), so the
held-out number is the whole point. A high held-out R^2 means the map is inductive:
a file that was never in the graph gets its per-relation position from its content
alone, with no re-run. A low one is also a finding — it means the relation's geometry
is not a linear function of content.
"""
import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS = "CheckItOutV3"
RELS = ["PERFORMS", "USES", "MODIFIES", "CALLS", "ACCESSES"]
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]
FOLDS = 5
SEED = 42


def fetch(driver, rel):
    """Nodes whose projection for `rel` is non-degenerate, with embedding and rho0."""
    q = f"""
    MATCH (n:EntityDetail {{namespace: $ns}})
    WHERE n.embedding IS NOT NULL AND n.rho0 IS NOT NULL AND n.`proj_{rel}` IS NOT NULL
      AND reduce(s = 0.0, x IN n.`proj_{rel}` | s + abs(x)) > 0.0
    RETURN n.embedding AS e, n.rho0 AS r0, n.`proj_{rel}` AS pk
    """
    with driver.session() as s:
        rows = list(s.run(q, ns=NS))
    X = np.array([r["e"] for r in rows], dtype=np.float64)
    R0 = np.array([r["r0"] for r in rows], dtype=np.float64)
    PK = np.array([r["pk"] for r in rows], dtype=np.float64)
    return X, PK, PK - R0  # features, target projection, target correction


def ridge_dual(X, Y, lam):
    """rho = X^T (X X^T + lam I)^-1 Y  — the n < d form. Returns (d, k)."""
    n = X.shape[0]
    K = X @ X.T
    A = np.linalg.solve(K + lam * np.eye(n), Y)
    return X.T @ A


def r2(Y, P):
    """Variance-weighted R^2 across output dimensions."""
    ss_res = float(((Y - P) ** 2).sum())
    ss_tot = float(((Y - Y.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def cross_validate(X, Y, lam, folds=FOLDS, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    parts = np.array_split(idx, folds)
    scores = []
    for f in range(folds):
        test = parts[f]
        train = np.concatenate([parts[g] for g in range(folds) if g != f])
        # centre on the training set only — no leakage
        mu_x, mu_y = X[train].mean(axis=0), Y[train].mean(axis=0)
        rho = ridge_dual(X[train] - mu_x, Y[train] - mu_y, lam)
        pred = (X[test] - mu_x) @ rho + mu_y
        scores.append(r2(Y[test], pred))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    print(f"{'relation':<11} {'n':>5} {'target':<11} {'best lam':>9} {'held-out R2':>12} {'sd':>7}")
    print("-" * 62)
    for rel in RELS:
        X, PK, DK = fetch(driver, rel)
        if len(X) < FOLDS * 2:
            print(f"{rel:<11} {len(X):>5}  too few nodes")
            continue
        for name, Y in (("projection", PK), ("correction", DK)):
            best = max(((cross_validate(X, Y, l), l) for l in LAMBDAS), key=lambda t: t[0][0])
            (mean, sd), lam = best
            print(f"{rel:<11} {len(X):>5} {name:<11} {lam:>9g} {mean:>12.3f} {sd:>7.3f}")
    driver.close()


if __name__ == "__main__":
    main()
