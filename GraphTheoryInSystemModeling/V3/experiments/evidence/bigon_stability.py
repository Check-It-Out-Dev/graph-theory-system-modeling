"""Is the ALGEBRA_VIOLATION anomaly real, or two fits' worth of noise?

The bigon table says ALGEBRA_VIOLATION sits ~74 degrees away from every relation
it shares a file pair with, while ordinary pairs sit at 5-14 degrees, and that
every one of its 342 bigons reverses orientation. Both numbers are derived from
the same two fitted maps, so "198 of 198 reversing" is not 198 observations -- it
is the sign of one determinant, counted 198 times. The determinant bootstrap
already showed that sign is a coin flip at 56%.

So: bootstrap the relation maps themselves, and report the rotation between each
pair as a distribution rather than a point. A pair is genuinely far apart only if
the 5th percentile of its bootstrap is still large.

Also reports the degree correlation of the low eigenvectors, because a low
spectrum that merely ranks hubs is not resolving sub-topologies.
"""
import collections

import numpy as np
import networkx as nx
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS = "CheckItOutV3"
D, N_BOOT, SEED = 8, 300, 42
FOCUS = ["ALGEBRA_VIOLATION", "IMPLEMENTS", "EXTENDS", "IMPORTS",
         "INJECTS", "PERFORMS", "USES", "MODIFIES", "CALLS", "ACCESSES"]


def procrustes(A, B):
    Ac, Bc = A - A.mean(axis=0), B - B.mean(axis=0)
    U, _, Vt = np.linalg.svd(Bc.T @ Ac)
    return U @ Vt


def rotation(P, Q):
    """Mean rotation angle of Q^T P, in degrees: how differently the two maps act."""
    H = Q.T @ P
    return float(np.degrees(np.mean(np.abs(np.angle(np.linalg.eigvals(H))))))


def main():
    rng = np.random.default_rng(SEED)
    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with driver.session() as s:
            nodes = list(s.run("MATCH (n:EntityDetail {namespace:$ns}) "
                               "WHERE n.rho0 IS NOT NULL RETURN id(n) AS id, n.rho0 AS x", ns=NS))
            edges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "WHERE a.rho0 IS NOT NULL AND b.rho0 IS NOT NULL "
                "RETURN id(a) AS s, id(b) AS t, type(r) AS k", ns=NS))
    finally:
        driver.close()

    idx = {r["id"]: i for i, r in enumerate(nodes)}
    X = np.array([r["x"] for r in nodes], dtype=np.float64)
    by = collections.defaultdict(list)
    for r in edges:
        by[r["k"]].append((idx[r["s"]], idx[r["t"]]))
    rels = [k for k in FOCUS if len(by[k]) >= D]

    # bootstrap every map once, reuse the draws for every pair
    draws = {}
    for k in rels:
        pairs = by[k]
        src = X[[p[0] for p in pairs]]
        tgt = X[[p[1] for p in pairs]]
        ds = []
        for _ in range(N_BOOT):
            take = rng.integers(0, len(pairs), len(pairs))
            ds.append(procrustes(src[take], tgt[take]))
        draws[k] = (procrustes(src, tgt), ds)

    print("BOOTSTRAPPED ROTATION BETWEEN RELATION MAPS")
    print("  a pair is genuinely far apart only if its 5th percentile is still large\n")
    print(f"{'pair':<40}{'point':>8}{'p5':>8}{'p50':>8}{'p95':>8}   verdict")

    anomalous = []
    for i, a in enumerate(rels):
        for b in rels[i + 1:]:
            pt = rotation(draws[a][0], draws[b][0])
            bs = np.array([rotation(pa, pb) for pa, pb in zip(draws[a][1], draws[b][1])])
            p5, p50, p95 = np.percentile(bs, [5, 50, 95])
            verdict = "FAR" if p5 > 45 else ("near" if p95 < 30 else "unresolved")
            if verdict == "FAR":
                anomalous.append((a, b, p5))
            if pt > 40 or p5 > 30 or (a == "ALGEBRA_VIOLATION" or b == "ALGEBRA_VIOLATION"):
                print(f"{a + ' / ' + b:<40}{pt:>7.1f}°{p5:>7.1f}°{p50:>7.1f}°{p95:>7.1f}°   {verdict}")

    print(f"\n  pairs separated beyond bootstrap noise: {len(anomalous)}")
    who = collections.Counter()
    for a, b, _ in anomalous:
        who[a] += 1
        who[b] += 1
    for k, c in who.most_common():
        print(f"    {k:<22} in {c} of them")

    # how far is each relation from the CROWD, not from one partner
    print("\n\nDISTANCE FROM THE CONSENSUS\n")
    print("  median rotation of each relation against all the others\n")
    print(f"{'relation':<22}{'edges':>7}{'median vs rest':>16}")
    rows = []
    for a in rels:
        vs = [rotation(draws[a][0], draws[b][0]) for b in rels if b != a]
        rows.append((a, len(by[a]), float(np.median(vs))))
    for a, m, med in sorted(rows, key=lambda r: -r[2]):
        print(f"{a:<22}{m:>7}{med:>15.1f}°")

    print("\n\nDEGREE CHECK ON THE LOW SPECTRUM\n")
    st = np.load("sheaf_state.npz", allow_pickle=True)
    G = nx.Graph()
    G.add_edges_from((idx[r["s"]], idx[r["t"]]) for r in edges
                     if r["k"] in rels and idx[r["s"]] != idx[r["t"]])
    giant = sorted(max(nx.connected_components(G), key=len))
    dgr = np.array([G.degree(v) for v in giant], dtype=float)
    vecs = st["vecs"]
    if vecs.shape[0] == len(giant) * D:
        print(f"{'eigenvector':<14}{'corr with degree':>18}")
        for i in range(min(8, vecs.shape[1])):
            mass = (vecs[:, i].reshape(len(giant), D) ** 2).sum(axis=1)
            print(f"  phi_{i:<10}{np.corrcoef(dgr, mass)[0, 1]:>+17.3f}")
        print("\n  r near +1 means that eigenvector is ranking hubs, not resolving a sub-topology")
    else:
        print(f"  saved eigenvectors ({vecs.shape}) do not match this node set "
              f"({len(giant)} x {D}); rerun sheaf_laplacian.py first")


if __name__ == "__main__":
    main()
