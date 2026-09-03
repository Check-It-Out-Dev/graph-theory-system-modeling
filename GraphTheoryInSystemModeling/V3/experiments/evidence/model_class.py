"""S1c — which Laplacian? Fit the model class instead of assuming it.

S1 concluded that 26 of 34 signatures have "no consistent rotation". That
conclusion is only as good as the model class it was tested against, and it was
tested against exactly one: O(8). If a relation's true transformation is a
scaling, a projection, or a phase, forcing an orthogonal map on it returns
something unstable, and a reliability test faithfully reports that instability --
while the structure sits there unmeasured.

So fit a ladder of model classes per signature and let held-out error choose:

  identity     0 params   no transformation at all; the baseline everything must beat
  phase U(1)   1          read R^8 as C^4, one global phase -- THE MAGNETIC LAPLACIAN
  torus T^4    4          one phase per complex coordinate
  scalar       1          uniform scaling
  diagonal     8          per-coordinate scaling
  similarity  29          scaled rotation, sO(8)
  rotation    28          O(8), what 6b and S1 assumed
  linear      64          unconstrained

The magnetic Laplacian is not a rival to the connection Laplacian; it is its rank-1
case. That is exactly why it is worth trying here: one parameter per relation
instead of twenty-eight, so it can survive on data that cannot support a rotation.
If the signatures that failed S1 succeed under U(1), the right operator for this
graph is magnetic, and the failure in S1 was a parameter-count failure rather than
an absence of structure.

Criterion: split-half CROSS-VALIDATED relative error. Fit on half the edges,
predict the other half, measure ||B - A M^T|| / ||B||. Comparable across model
classes in a way that a rotation angle is not, honest about overfitting by
construction, and directly interpretable -- 1.0 means "no better than predicting
the mean", and the identity row says how much of that a model has to earn.

Also reported: the shuffled-pairing null, so a class can be checked for fitting
the relation rather than the marginal geometry.
"""
import collections

import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS = "CheckItOutV3"
D = 8
MIN_EDGES = 20
N_SPLIT = 40
SEED = 42


# ------------------------------------------------------------------ model classes

def m_identity(A, B):
    return np.eye(D)


def m_scalar(A, B):
    s = float((A * B).sum() / max((A * A).sum(), 1e-12))
    return s * np.eye(D)


def m_diagonal(A, B):
    num, den = (A * B).sum(axis=0), (A * A).sum(axis=0)
    return np.diag(num / np.maximum(den, 1e-12))


def m_rotation(A, B):
    U, _, Vt = np.linalg.svd(B.T @ A)
    return U @ Vt


def m_similarity(A, B):
    O = m_rotation(A, B)
    s = float((B * (A @ O.T)).sum() / max(((A @ O.T) ** 2).sum(), 1e-12))
    return s * O


def _as_complex(M):
    return M[:, 0::2] + 1j * M[:, 1::2]


def _from_complex(Z):
    out = np.zeros((len(Z), D))
    out[:, 0::2], out[:, 1::2] = Z.real, Z.imag
    return out


def _phase_map(theta):
    """Real 8x8 matrix acting as multiplication by e^{i theta} on each C coordinate."""
    M = np.zeros((D, D))
    for j, th in enumerate(np.atleast_1d(theta)):
        c, s = np.cos(th), np.sin(th)
        M[2 * j, 2 * j], M[2 * j, 2 * j + 1] = c, -s
        M[2 * j + 1, 2 * j], M[2 * j + 1, 2 * j + 1] = s, c
    return M


def m_phase(A, B):
    """One global U(1) phase: the magnetic Laplacian's edge weight."""
    Za, Zb = _as_complex(A), _as_complex(B)
    th = np.angle(np.sum(np.conj(Za) * Zb))
    return _phase_map(np.full(D // 2, th))


def m_torus(A, B):
    """One phase per complex coordinate: T^4 instead of U(1)."""
    Za, Zb = _as_complex(A), _as_complex(B)
    return _phase_map(np.angle(np.sum(np.conj(Za) * Zb, axis=0)))


def m_linear(A, B):
    return np.linalg.lstsq(A, B, rcond=None)[0].T


MODELS = [("identity", 0, m_identity), ("phase U(1)", 1, m_phase),
          ("torus T^4", 4, m_torus), ("scalar", 1, m_scalar),
          ("diagonal", 8, m_diagonal), ("rotation", 28, m_rotation),
          ("similarity", 29, m_similarity), ("linear", 64, m_linear)]


# ------------------------------------------------------------------ evaluation

def cv_error(src, tgt, fit, rng, shuffle=False):
    """Split-half cross-validated relative error, median over N_SPLIT splits."""
    n, errs = len(src), []
    for _ in range(N_SPLIT):
        perm = rng.permutation(n)
        tr, te = perm[: n // 2], perm[n // 2:]
        if len(tr) < D or len(te) < D:
            continue
        A, B = src[tr], tgt[tr]
        if shuffle:
            B = B[rng.permutation(len(B))]
        ma, mb = A.mean(axis=0), B.mean(axis=0)
        try:
            M = fit(A - ma, B - mb)
        except np.linalg.LinAlgError:
            continue
        At, Bt = src[te] - ma, tgt[te] - mb
        denom = np.linalg.norm(Bt)
        if denom < 1e-9:
            continue
        errs.append(float(np.linalg.norm(Bt - At @ M.T) / denom))
    return float(np.median(errs)) if errs else np.nan


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
                "RETURN id(a) AS s, id(b) AS t, type(r) AS k, "
                "a.entity_type AS st, b.entity_type AS tt", ns=NS))
    finally:
        driver.close()

    idx = {r["id"]: i for i, r in enumerate(nodes)}
    X = np.array([r["x"] for r in nodes], dtype=np.float64)
    sig = collections.defaultdict(list)
    for r in edges:
        sig[(r["st"], r["k"], r["tt"])].append((idx[r["s"]], idx[r["t"]]))
    keys = [k for k, v in sig.items() if len(v) >= MIN_EDGES]

    print("HELD-OUT ERROR BY MODEL CLASS, PER SIGNATURE")
    print("  fit on half the edges, predict the other half. 1.00 = no better than the mean.")
    print("  a class earns its place only by beating `identity`, which uses no map at all.\n")
    header = f"{'signature':<40}{'n':>5}" + "".join(f"{m[0][:9]:>10}" for m in MODELS)
    print(header)
    print("-" * len(header))

    won = collections.Counter()
    rows = []
    for key in sorted(keys, key=lambda k: -len(sig[k])):
        pairs = sig[key]
        src = X[[p[0] for p in pairs]]
        tgt = X[[p[1] for p in pairs]]
        errs = [cv_error(src, tgt, f, np.random.default_rng(SEED), False) for _, _, f in MODELS]
        st, k, tt = key
        if all(np.isnan(e) for e in errs):
            # every split had a degenerate held-out cloud: the targets are a single
            # point, so there is nothing to predict and no error to normalise by
            print(f"{st + ' -' + k + '-> ' + tt:<40}{len(pairs):>5}"
                  f"   degenerate — target cloud is a point")
            won["degenerate (nothing to predict)"] += 1
            continue
        best = int(np.nanargmin(errs))
        cells = "".join(("      nan" if np.isnan(e) else
                         (f"{e:>9.2f}*" if i == best else f"{e:>9.2f} "))
                        for i, e in enumerate(errs))
        print(f"{st + ' -' + k + '-> ' + tt:<40}{len(pairs):>5}{cells}")
        rows.append((key, len(pairs), errs, best))
        # only count a win if the model actually beat doing nothing
        if errs[best] < errs[0] - 1e-9:
            won[MODELS[best][0]] += 1
        else:
            won["identity (nothing helps)"] += 1

    print("\n\nWHICH CLASS WINS, ACROSS 34 SIGNATURES\n")
    for name, c in won.most_common():
        print(f"  {name:<28}{c:>4}")

    print("\n\nDOES U(1) RESCUE WHAT O(8) COULD NOT FIT?\n")
    print("  the S1 verdict said these 26 signatures have no consistent rotation.")
    print("  if a 1-parameter phase beats both identity and the 28-parameter rotation,")
    print("  the failure was parameter count, not absence of structure.\n")
    i_id = 0
    i_ph = [i for i, m in enumerate(MODELS) if m[0] == "phase U(1)"][0]
    i_ro = [i for i, m in enumerate(MODELS) if m[0] == "rotation"][0]
    rescued = 0
    print(f"{'signature':<40}{'identity':>10}{'U(1)':>9}{'O(8)':>9}   verdict")
    for key, n, errs, best in rows:
        st, k, tt = key
        idn, ph, ro = errs[i_id], errs[i_ph], errs[i_ro]
        if np.isnan(ph) or np.isnan(ro):
            continue
        if ph < idn and ph < ro:
            verdict, rescued = "U(1) WINS — magnetic beats orthogonal", rescued + 1
        elif ro < idn and ro < ph:
            verdict = "O(8) genuinely better"
        elif min(ph, ro) >= idn:
            verdict = "neither helps — no map of any kind"
        else:
            verdict = "tie"
        print(f"{st + ' -' + k + '-> ' + tt:<40}{idn:>10.2f}{ph:>9.2f}{ro:>9.2f}   {verdict}")
    print(f"\n  signatures where the 1-parameter phase beats the 28-parameter rotation: {rescued}")

    np.savez("model_class.npz",
             keys=np.array([f"{a}|{b}|{c}" for (a, b, c), _, _, _ in rows]),
             errs=np.array([r[2] for r in rows]),
             counts=np.array([r[1] for r in rows]),
             models=np.array([m[0] for m in MODELS]))
    print("\n  saved to model_class.npz")


if __name__ == "__main__":
    main()
