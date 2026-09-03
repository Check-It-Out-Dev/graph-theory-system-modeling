"""
A connection sheaf over the code graph, and the Laplacian that goes with it.

The earlier holonomy experiment composed transition maps around loops in RELATION
space -- PERFORMS -> USES -> MODIFIES -> PERFORMS. That is a real measurement, but
it throws the graph away: the answer is one number per triple of relation types,
and it cannot say WHERE in the codebase the phase lives.

This puts the connection back on the graph. Each node carries a stalk R^8 (its
position in the base topology). Each relation type k carries one orthogonal map
O_k in O(8): "to compare the two ends of a k-edge, first rotate by O_k". Then

  * the DIRICHLET ENERGY  sum ||O_k x_u - x_v||^2  measures how badly the typed
    structure fails to agree with the geometry;

  * the CONNECTION LAPLACIAN  L = D (x) I - (O_k blocks)  is the operator whose
    kernel is the set of globally consistent assignments, H^0 of the sheaf. If the
    connection is flat, dim H^0 = 8 per component. Every dimension MISSING from
    that is a dimension the codebase cannot frame consistently -- frustration, as
    an integer;

  * its LOW EIGENVECTORS give soft sub-topology membership. Node v belongs to
    sub-topology i with weight ||phi_i(v)||^2. This is the answer to "which nodes
    are in which sub-topology": not a bitmask over supports, but mass of a
    near-section. Supports say where a relation is DEFINED; sections say where the
    geometry is CONSISTENT, which is the thing worth knowing;

  * HOLONOMY AROUND CYCLES IN THE GRAPH, not in relation space. Take a spanning
    tree, take the fundamental cycles, and for each one multiply the O's around
    it. The rotation angle is that cycle's Berry phase. Attributing it back to the
    nodes and edges on the cycle gives a per-node curvature -- which is exactly
    "which areas of the graph carry the phase".

  * and the Z/2 invariant, now exact. Procrustes returns a genuine element of
    O(8), so det(holonomy) = +-1 with no numerical hedging. A cycle with det = -1
    admits no consistent orientation. If any exists, the code graph has no global
    orientation, full stop -- a topological obstruction, not an analogy.

Why orthogonal Procrustes and not least squares: the earlier lstsq fits reached
condition number 442, meaning those maps were collapsing directions rather than
moving them, so "det < 0" was measuring numerical mush. Constraining to O(8) makes
the fit closed-form (Kabsch), makes composition a group operation, and makes every
invariant below exact. The price is that a relation whose geometry is genuinely
non-rigid now shows up as a large RESIDUAL instead of a distorted matrix -- which
is the honest place for it to show up.

Null model: shuffle which target pairs with which source inside each relation,
refit, recompute. If O_k is picking up structure, the true residual must beat it.
"""
import collections
import itertools
import sys

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS = "CheckItOutV3"
D = 8                       # stalk dimension
MIN_EDGES = D               # below this an O(8) fit is underdetermined
N_PERM = 20
N_BOOT = 200
N_EIG = 24
SEED = 42


# ---------------------------------------------------------------- loading

def load(driver):
    with driver.session() as s:
        nodes = list(s.run(
            "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.rho0 IS NOT NULL "
            "RETURN id(n) AS id, n.name AS name, n.path AS path, n.rho0 AS x",
            ns=NS))
        edges = list(s.run(
            "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
            "WHERE a.rho0 IS NOT NULL AND b.rho0 IS NOT NULL "
            "RETURN id(a) AS s, id(b) AS t, type(r) AS k",
            ns=NS))
    idx = {r["id"]: i for i, r in enumerate(nodes)}
    X = np.array([r["x"] for r in nodes], dtype=np.float64)
    names = [r["name"] for r in nodes]
    paths = [r["path"] for r in nodes]
    E = [(idx[r["s"]], idx[r["t"]], r["k"]) for r in edges]
    return idx, names, paths, X, E


# ---------------------------------------------------------------- the connection

def procrustes(A, B):
    """Nearest O in O(d) with O @ a ~ b, and the mean residual it leaves.

    Kabsch: centre both clouds, SVD the cross-covariance, O = U V^T. Closed form,
    exactly orthogonal, so det(O) = +-1 is meaningful rather than approximate.
    """
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    Ac, Bc = A - ca, B - cb
    U, _, Vt = np.linalg.svd(Bc.T @ Ac)
    O = U @ Vt
    # normalise by the pooled spread of both clouds, not the target's alone: a
    # relation whose targets are geometrically degenerate (all test files sit at
    # nearly one point, having no structure but the test edge) otherwise divides
    # by ~0 and reports a residual of 1e16 instead of "these are degenerate".
    scale = np.sqrt(max(np.linalg.norm(Ac) * np.linalg.norm(Bc), 1e-24))
    resid = np.linalg.norm(Ac @ O.T - Bc) / scale
    return O, float(resid), float(np.linalg.norm(Bc) / max(np.sqrt(len(B)), 1))


def fit_connection(X, E, rng):
    """One O_k per relation type, with a shuffled-pairing null and a bootstrap
    over the determinant sign — orientation is a claim about a discrete invariant,
    so it has to be shown stable under resampling before it means anything."""
    by_type = collections.defaultdict(list)
    for s, t, k in E:
        by_type[k].append((s, t))

    conn, report, skipped = {}, [], []
    for k, pairs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        if len(pairs) < MIN_EDGES:
            skipped.append((k, len(pairs)))
            continue
        src = X[[p[0] for p in pairs]]
        tgt = X[[p[1] for p in pairs]]
        O, resid, spread = procrustes(src, tgt)
        det = float(np.linalg.det(O))

        nulls = []
        for _ in range(N_PERM):
            perm = rng.permutation(len(pairs))
            nulls.append(procrustes(src, tgt[perm])[1])
        nm, nsd = float(np.mean(nulls)), float(np.std(nulls))
        z = (resid - nm) / nsd if nsd > 1e-9 else np.nan

        agree = 0
        for _ in range(N_BOOT):
            take = rng.integers(0, len(pairs), len(pairs))
            Ob, _, _ = procrustes(src[take], tgt[take])
            agree += int(np.sign(np.linalg.det(Ob)) == np.sign(det))
        conn[k] = O
        report.append((k, len(pairs), resid, nm, z, det, agree / N_BOOT, spread))
    return conn, report, skipped


# ---------------------------------------------------------------- the Laplacian

def connection_laplacian(nodes, E, conn):
    """Symmetric normalised connection Laplacian over `nodes`, sparse.

    Edge (u,v,k) contributes  -O_k  to block (v,u) and  -O_k^T  to block (u,v),
    which makes L Hermitian; degrees count the edges actually used.

    Restricted to a node subset on purpose. A disconnected component contributes
    its own full copy of the kernel, and an ISOLATED node contributes all of R^8,
    so running this over the whole graph makes dim H^0 a count of loose parts
    rather than a measure of frustration. The question worth asking is per
    component, and really about the giant one.
    """
    loc = {v: i for i, v in enumerate(nodes)}
    m = len(nodes)
    used = [(u, v, k) for u, v, k in E
            if k in conn and u != v and u in loc and v in loc]

    deg = np.zeros(m)
    for u, v, _ in used:
        deg[loc[u]] += 1
        deg[loc[v]] += 1
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)

    rows, cols, vals = [], [], []
    for i in range(m):
        if deg[i] > 0:
            for a in range(D):
                rows.append(i * D + a); cols.append(i * D + a); vals.append(1.0)
    for u, v, k in used:
        i, j = loc[u], loc[v]
        O = conn[k] * (inv[i] * inv[j])
        for a in range(D):
            for b in range(D):
                if O[a, b]:
                    rows.append(j * D + a); cols.append(i * D + b); vals.append(-O[a, b])
                    rows.append(i * D + b); cols.append(j * D + a); vals.append(-O[a, b])
    return sp.coo_matrix((vals, (rows, cols)), shape=(m * D, m * D)).tocsr(), deg


# ---------------------------------------------------------------- holonomy on the graph

def bigon_holonomy(E, conn, names):
    """The shortest cycles there are: two files joined by two DIFFERENT relations.

    If u -k1-> v and u -k2-> v both hold, the loop out along k1 and back along k2
    has holonomy O_k2^T O_k1, and it needs no spanning tree and no path. It asks
    the sharpest possible version of the question: when two relation types connect
    the same pair of files, do they place them in the same relative position?

    These are dropped by any simple-graph construction, and they are the most
    informative cycles in the graph.
    """
    pair = collections.defaultdict(set)
    for u, v, k in E:
        if k in conn and u != v:
            pair[(u, v)].add(k)

    out = collections.defaultdict(list)
    for (u, v), ks in pair.items():
        for k1, k2 in itertools.combinations(sorted(ks), 2):
            H = conn[k2].T @ conn[k1]
            rot = float(np.degrees(np.mean(np.abs(np.angle(np.linalg.eigvals(H))))))
            out[(k1, k2)].append((rot, float(np.linalg.det(H)), names[u], names[v]))
    return out


def cycle_holonomy(n, E, conn, rng, max_cycles=4000):
    """Fundamental cycles of a spanning tree; holonomy is the product around each."""
    G = nx.Graph()
    edge_map = {}
    for u, v, k in E:
        if k in conn and u != v:
            G.add_edge(u, v)
            edge_map.setdefault((min(u, v), max(u, v)), []).append((u, v, k))

    if G.number_of_edges() == 0:
        return [], np.zeros(n)

    tree = nx.minimum_spanning_tree(G)
    tree_paths = {}
    results = []
    node_phase = collections.defaultdict(list)

    extra = [e for e in G.edges() if not tree.has_edge(*e)]
    rng.shuffle(extra)
    for (a, b) in extra[:max_cycles]:
        try:
            path = nx.shortest_path(tree, a, b)
        except nx.NetworkXNoPath:
            continue
        walk = list(zip(path, path[1:])) + [(b, a)]

        H = np.eye(D)
        ok = True
        for (u, v) in walk:
            key = (min(u, v), max(u, v))
            cands = edge_map.get(key)
            if not cands:
                ok = False
                break
            su, sv, k = cands[0]
            O = conn[k] if (su, sv) == (u, v) else conn[k].T
            H = O @ H
        if not ok:
            continue

        ev = np.linalg.eigvals(H)
        rot = float(np.degrees(np.mean(np.abs(np.angle(ev)))))
        det = float(np.linalg.det(H))
        results.append((len(walk), rot, det))
        for v in path:
            node_phase[v].append(rot)

    curv = np.zeros(n)
    for v, vals in node_phase.items():
        curv[v] = float(np.mean(vals))
    return results, curv


# ---------------------------------------------------------------- report

def main():
    rng = np.random.default_rng(SEED)
    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        idx, names, paths, X, E = load(driver)
    finally:
        driver.close()
    n = len(X)
    print(f"loaded {n} nodes with a base position, {len(E)} typed edges\n")

    print("1. THE CONNECTION — one orthogonal map per relation, fitted by Procrustes\n")
    conn, report, skipped = fit_connection(X, E, rng)
    print(f"{'relation':<20}{'edges':>7}{'resid':>8}{'null':>8}{'z':>8}"
          f"{'spread':>9}{'det':>6}{'stable':>8}  class")
    print("  resid: 0 = the rotation explains the relation exactly, ~1 = no better than the mean")
    print("  stable: fraction of 200 bootstraps agreeing on the sign of det")
    for k, m, resid, nm, z, det, stab, spread in report:
        cls = "solid" if det > 0 else "REVERSED"
        if spread < 1e-3:
            cls += " (degenerate targets)"
        print(f"{k:<20}{m:>7}{resid:>8.3f}{nm:>8.3f}{z:>8.1f}"
              f"{spread:>9.4f}{det:>6.0f}{stab:>8.0%}  {cls}")
    if skipped:
        print("\n  too sparse for an O(8) fit, excluded: "
              + ", ".join(f"{k} ({m})" for k, m in skipped))

    print("\n\n2. THE LAPLACIAN — is there one global frame for the codebase?\n")
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((u, v) for u, v, k in E if k in conn and u != v)
    comps = sorted((c for c in nx.connected_components(G) if len(c) > 1),
                   key=len, reverse=True)
    isolated = int(sum(1 for c in nx.connected_components(G) if len(c) == 1))
    trees = sum(1 for c in comps if G.subgraph(c).number_of_edges() == len(c) - 1)
    print(f"  nodes with no typed edge at all : {isolated}  (each contributes a free R^8;")
    print(f"                                     counting them makes H^0 a parts count, not a measure)")
    print(f"  components of size > 1          : {len(comps)}  of which {trees} are trees")
    print(f"                                     (a tree has no cycles, so it is flat by construction)")
    print(f"  giant component                 : {len(comps[0])} nodes, "
          f"{G.subgraph(comps[0]).number_of_edges()} edges\n")

    giant = sorted(comps[0])
    L, deg = connection_laplacian(giant, E, conn)
    kk = min(N_EIG, len(giant) * D - 2)
    vals, vecs = spl.eigsh(L, k=kk, sigma=-1e-6, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    tol = 1e-8
    h0 = int((vals < tol).sum())
    print(f"  on the giant component, a FLAT connection would give dim H^0 = {D}")
    print(f"  measured dim H^0 (eigenvalues < {tol:g})                      = {h0}")
    print(f"  frustration: dimensions with no global frame                = {D - h0}")
    print("\n  lowest eigenvalues:  " + "  ".join(f"{v:.5f}" for v in vals[:10]))
    print(f"  spectral gap to the first non-section: {vals[h0]:.5f}"
          if h0 < kk else "")

    print("\n\n3. SOFT SUB-TOPOLOGY MEMBERSHIP — mass of each near-section\n")
    print("  a node belongs to sub-topology i with weight ||phi_i(v)||^2.")
    print("  supports say where a relation is DEFINED; these say where it is CONSISTENT.")
    gnames = [names[v] for v in giant]
    for i in range(min(5, kk)):
        mass = (vecs[:, i].reshape(len(giant), D) ** 2).sum(axis=1)
        top = np.argsort(-mass)[:7]
        share = mass[top].sum() / max(mass.sum(), 1e-12)
        kind = "section (exact)" if vals[i] < tol else "near-section"
        print(f"\n  {kind} {i}  (lambda = {vals[i]:.5f}, top 7 hold {share:.1%} of the mass)")
        for t in top:
            if mass[t] > 1e-9:
                print(f"      {mass[t]:>7.4f}  {str(gnames[t])[:60]}")

    print("\n\n3b. BIGONS — two files joined by two different relation types\n")
    bi = bigon_holonomy(E, conn, names)
    if bi:
        print(f"{'relation pair':<38}{'n':>5}{'median rot':>12}{'reversing':>11}")
        for (k1, k2), rows in sorted(bi.items(), key=lambda kv: -len(kv[1])):
            rots = np.array([r[0] for r in rows])
            rev = sum(1 for r in rows if r[1] < 0)
            print(f"{k1 + ' / ' + k2:<38}{len(rows):>5}{np.median(rots):>11.1f}°{rev:>11}")
        worst = max(((r, kp) for kp, rows in bi.items() for r in rows),
                    key=lambda t: t[0][0])
        print(f"\n  sharpest disagreement: {worst[1][0]} vs {worst[1][1]} at {worst[0][0]:.1f}°")
        print(f"    {worst[0][2]}  ->  {worst[0][3]}")
    else:
        print("  no file pair carries two different relation types")

    print("\n\n4. HOLONOMY ON THE GRAPH — Berry phase per cycle, attributed to nodes\n")
    cycles, curv = cycle_holonomy(n, E, conn, rng)
    if cycles:
        rots = np.array([c[1] for c in cycles])
        dets = np.array([c[2] for c in cycles])
        lens = np.array([c[0] for c in cycles])
        rev = int((dets < 0).sum())
        print(f"  independent cycles sampled : {len(cycles)}")
        print(f"  cycle length               : median {np.median(lens):.0f}, max {lens.max()}")
        print(f"  rotation (degrees)         : median {np.median(rots):.1f}, "
              f"mean {rots.mean():.1f}, max {rots.max():.1f}")
        print(f"  orientation-reversing      : {rev} of {len(cycles)} ({rev/len(cycles):.1%})")
        print(f"\n  => the code graph {'has NO' if rev else 'HAS a'} consistent global orientation")

        print("\n  files carrying the most phase (highest mean cycle rotation through them):")
        for v in np.argsort(-curv)[:12]:
            if curv[v] > 0:
                print(f"      {curv[v]:>6.1f}°  {str(names[v])[:62]}")

        print("\n  files carrying the least (they sit inside flat regions):")
        flat = [v for v in np.argsort(curv) if curv[v] > 0]
        for v in flat[:8]:
            print(f"      {curv[v]:>6.1f}°  {str(names[v])[:62]}")
    else:
        print("  no cycles found")

    np.savez("sheaf_state.npz",
             curv=curv, vals=vals, vecs=vecs,
             names=np.array([str(x) for x in names]),
             paths=np.array([str(x) for x in paths]),
             rels=np.array(sorted(conn)),
             conn=np.array([conn[k] for k in sorted(conn)]))
    print("\n  state saved to sheaf_state.npz")


if __name__ == "__main__":
    sys.exit(main())
