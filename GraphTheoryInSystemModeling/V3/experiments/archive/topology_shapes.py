"""
The shapes of the sub-topologies, and what happens going around a loop.

Four questions, in the order they have to be answered:

1. BUBBLES. Each relation resolves a sub-topology, but only over the nodes it
   actually touches. Which parts of the graph are shared between bubbles, and
   which belong to exactly one relation?

2. BACK IN THE ORIGINAL SPACE. Each fitted rho_k is a 4096x8 matrix, so its
   column space is an 8-dimensional subspace of the original embedding space --
   the directions relation k reads. Principal angles between those subspaces say
   which relations look at shared directions and which look at unique ones. This
   is the overlap question asked where it actually lives, in R^4096, not in the
   8-dimensional shadows.

3. SHAPE. What does each point cloud look like -- round, flat, filamentary? The
   eigenvalue spectrum of its covariance gives the aspect ratios, and a TwoNN
   estimate gives the local intrinsic dimension, which is the stratification
   claim from the foundations paper.

4. HOLONOMY. Fit the transition map M_ij taking positions in topology i to
   positions in topology j, compose around a closed loop of relations, and see
   whether you come back to where you started. Deviation from the identity is
   the discrete analogue of a Berry phase.

   And the homotopy question in its honest form: GL(8,R) has exactly two
   connected components, separated by the sign of the determinant. A transition
   map with det > 0 can be deformed continuously into the identity -- "solid".
   One with det < 0 cannot: it reverses orientation, and no continuous path of
   invertible maps connects it to doing nothing. That is a real topological
   invariant of the transformation, not a metaphor.
"""
import itertools
import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS = "CheckItOutV3"
RELS = ["PERFORMS", "USES", "MODIFIES", "CALLS", "ACCESSES"]
LAM = 0.1


def load(driver):
    props = ", ".join(f"n.`proj_{r}` AS {r}" for r in RELS)
    q = f"""
    MATCH (n:EntityDetail {{namespace: $ns}})
    WHERE n.embedding IS NOT NULL AND n.rho0 IS NOT NULL
    RETURN id(n) AS id, n.name AS name, n.embedding AS emb, n.rho0 AS rho0, {props}
    """
    with driver.session() as s:
        rows = list(s.run(q, ns=NS))
    ids = [r["id"] for r in rows]
    names = [r["name"] for r in rows]
    X = np.array([r["emb"] for r in rows], dtype=np.float64)
    R0 = np.array([r["rho0"] for r in rows], dtype=np.float64)
    P, support = {}, {}
    for rel in RELS:
        vals = [r[rel] for r in rows]
        M = np.array([v if v is not None else [0.0] * 8 for v in vals], dtype=np.float64)
        P[rel] = M
        support[rel] = np.abs(M).sum(axis=1) > 0
    return ids, names, X, R0, P, support


def bubbles(names, support):
    print("\n1. BUBBLES — which parts of the graph each relation touches\n")
    print(f"{'relation':<11} {'support':>8} {'exclusive':>10}")
    counts = np.sum([support[r] for r in RELS], axis=0)
    for rel in RELS:
        excl = int(np.sum(support[rel] & (counts == 1)))
        print(f"{rel:<11} {int(support[rel].sum()):>8} {excl:>10}")
    print(f"\n{'nodes in k bubbles':<22}{'count':>7}")
    for k in range(0, len(RELS) + 1):
        c = int(np.sum(counts == k))
        if c:
            print(f"  k = {k:<19}{c:>7}")

    print(f"\n{'pair':<24}{'shared':>8}{'jaccard':>9}")
    for a, b in itertools.combinations(RELS, 2):
        inter = int(np.sum(support[a] & support[b]))
        union = int(np.sum(support[a] | support[b]))
        print(f"{a + ' & ' + b:<24}{inter:>8}{inter / union if union else 0:>9.3f}")

    hubs = np.argsort(-counts)[:8]
    print("\n  files sitting in the most bubbles (the multi-role ones):")
    for i in hubs:
        if counts[i] > 1:
            roles = [r for r in RELS if support[r][i]]
            print(f"    {counts[i]}  {str(names[i])[:46]:<46} {','.join(roles)}")


def fit_rho(X, Y, mask, lam=LAM):
    """Ridge in dual form over the supported rows; returns (4096, 8)."""
    Xs, Ys = X[mask], Y[mask]
    mu = Xs.mean(axis=0)
    Xc, Yc = Xs - mu, Ys - Ys.mean(axis=0)
    K = Xc @ Xc.T
    return Xc.T @ np.linalg.solve(K + lam * np.eye(len(Xc)), Yc)


def principal_angles(A, B):
    """Angles in degrees between the column spaces of A and B."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def subspaces(X, P, support):
    print("\n\n2. BACK IN THE ORIGINAL SPACE — which directions of R^4096 each relation reads\n")
    rho = {r: fit_rho(X, P[r], support[r]) for r in RELS}
    print(f"{'pair':<24}{'shared dirs':>12}{'min angle':>11}{'mean angle':>12}")
    print("  (a direction counts as shared when its principal angle is under 30 degrees)")
    for a, b in itertools.combinations(RELS, 2):
        ang = principal_angles(rho[a], rho[b])
        print(f"{a + ' & ' + b:<24}{int(np.sum(ang < 30)):>12}{ang.min():>11.1f}{ang.mean():>12.1f}")
    return rho


def shapes(P, support):
    print("\n\n3. SHAPE — what each point cloud looks like\n")
    print(f"{'relation':<11}{'n':>6}{'eff.rank':>10}{'flatness':>10}{'local ID':>10}")
    print("  eff.rank = exp(entropy of the covariance spectrum); flatness = sigma_1 / sigma_8")
    for rel in RELS:
        Y = P[rel][support[rel]]
        if len(Y) < 12:
            print(f"{rel:<11}{len(Y):>6}   too few points")
            continue
        Yc = Y - Y.mean(axis=0)
        sv = np.linalg.svd(Yc, compute_uv=False)
        p = sv / sv.sum()
        eff = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        flat = float(sv[0] / max(sv[-1], 1e-12))

        # TwoNN maximum-likelihood intrinsic dimension
        D = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        srt = np.sort(D, axis=1)
        r1, r2 = srt[:, 0], srt[:, 1]
        ok = (r1 > 1e-12) & (r2 > r1)
        mu = r2[ok] / r1[ok]
        idim = float(len(mu) / np.log(mu).sum()) if len(mu) else float("nan")
        print(f"{rel:<11}{len(Y):>6}{eff:>10.2f}{flat:>10.1f}{idim:>10.2f}")


def transitions(P, support):
    print("\n\n4. HOLONOMY AND HOMOTOPY — going around a loop\n")
    M, shared = {}, {}
    for a, b in itertools.permutations(RELS, 2):
        m = support[a] & support[b]
        shared[(a, b)] = int(m.sum())
        if m.sum() >= 12:
            M[(a, b)] = np.linalg.lstsq(P[a][m], P[b][m], rcond=None)[0]

    print(f"{'transition':<24}{'n':>5}{'det':>10}{'class':>10}{'cond':>9}")
    print("  class: det > 0 is deformable to the identity; det < 0 reverses orientation")
    for (a, b), m in sorted(M.items()):
        d = float(np.linalg.det(m))
        cond = float(np.linalg.cond(m))
        cls = "solid" if d > 0 else "REVERSED"
        print(f"{a + ' -> ' + b:<24}{shared[(a, b)]:>5}{d:>10.3f}{cls:>10}{cond:>9.1f}")

    print(f"\n{'loop':<34}{'||H - I||_F':>12}{'rotation':>10}{'det':>9}")
    print("  a loop that returns you exactly where you started would read 0.000")
    for a, b, c in itertools.combinations(RELS, 3):
        keys = [(a, b), (b, c), (c, a)]
        if not all(k in M for k in keys):
            continue
        H = M[keys[0]] @ M[keys[1]] @ M[keys[2]]
        dev = float(np.linalg.norm(H - np.eye(8)) / np.sqrt(8))
        ev = np.linalg.eigvals(H)
        rot = float(np.degrees(np.mean(np.abs(np.angle(ev)))))
        print(f"{a + ' -> ' + b + ' -> ' + c + ' -> ' + a:<34}{dev:>12.3f}{rot:>9.1f}°{float(np.linalg.det(H)):>9.3f}")


def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        ids, names, X, R0, P, support = load(driver)
        print(f"loaded {len(ids)} nodes, embedding dim {X.shape[1]}")
        bubbles(names, support)
        subspaces(X, P, support)
        shapes(P, support)
        transitions(P, support)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
