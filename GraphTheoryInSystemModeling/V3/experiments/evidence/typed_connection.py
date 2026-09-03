"""S1 — does the is-a / uses split survive typing the nodes?

Section 6b found the relation set bimodal: EXTENDS, IMPLEMENTS and
ALGEBRA_VIOLATION sit 67-77 degrees from everything else, the other seven agree
within 13, and that reads as subtyping versus usage. But those same three
relations are the ones whose edges span MANY node-type signatures -- EXTENDS
appears as Resource->Resource, Rule->Resource, Actor->Resource and
Process->Resource -- while PERFORMS is always Actor->Process, USES and MODIFIES
always Process->Resource, CALLS always Process->Process. Fitting one rotation
across four different signatures can produce a map that matches none of them, and
that would look exactly like an outlier.

So refit per signature (src_type, relation, tgt_type) and ask two questions.

  Q1  Do a relation's OWN signatures agree with each other? If EXTENDS's four
      signature-maps are mutually consistent, the pooled map was fine and the
      outlier status is architecture. If they disagree among themselves, the
      pooled map was an average of incompatible things and the outlier status is
      an artifact of pooling.

  Q2  At the signature level, is there still a two-family split, and does it
      still fall on subtyping versus usage?

The measurement problem underneath both: a rotation between two fitted maps grows
when either map is noisy, and noise grows as edges fall. EXTENDS's signatures have
28-116 edges; PERFORMS has 171 in one. Comparing them naively rewards the
better-sampled relation. So every signature gets its own noise floor by SPLIT-HALF
RELIABILITY: split its edges at random, fit a map on each half, measure the
rotation between the halves, repeat. That is how far apart two maps land when they
are estimates of the SAME thing at THAT sample size. A pair of signatures is
genuinely different only when the rotation between them exceeds both of their own
noise floors -- which controls for sample size automatically, without a correction
factor anybody has to justify.
"""
import collections
import itertools

import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS = "CheckItOutV3"
D = 8
MIN_EDGES = 20          # need >= 2*D so each split half can still fit an O(8)
N_SPLIT = 60
SEED = 42

SUBTYPING = {"EXTENDS", "IMPLEMENTS"}
USAGE = {"PERFORMS", "USES", "MODIFIES", "CALLS", "ACCESSES", "IMPORTS", "INJECTS"}


def procrustes(A, B):
    Ac, Bc = A - A.mean(axis=0), B - B.mean(axis=0)
    U, _, Vt = np.linalg.svd(Bc.T @ Ac)
    return U @ Vt


def rotation(P, Q):
    """Mean rotation angle of Q^T P in degrees: how differently the two maps act."""
    return float(np.degrees(np.mean(np.abs(np.angle(np.linalg.eigvals(Q.T @ P))))))


def family(rel):
    return "is-a" if rel in SUBTYPING else ("uses" if rel in USAGE else "other")


def load():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with driver.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.rho0 IS NOT NULL "
                "RETURN id(n) AS id, n.entity_type AS t, n.rho0 AS x", ns=NS))
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
    return X, sig


def fit_signatures(X, sig, rng):
    """One map per signature, plus its own split-half noise floor."""
    out = {}
    for key, pairs in sig.items():
        if len(pairs) < MIN_EDGES:
            continue
        src = X[[p[0] for p in pairs]]
        tgt = X[[p[1] for p in pairs]]
        O = procrustes(src, tgt)

        floors = []
        n = len(pairs)
        for _ in range(N_SPLIT):
            perm = rng.permutation(n)
            a, b = perm[: n // 2], perm[n // 2:]
            if len(a) < D or len(b) < D:
                continue
            floors.append(rotation(procrustes(src[a], tgt[a]),
                                   procrustes(src[b], tgt[b])))
        out[key] = (O, len(pairs), float(np.median(floors)) if floors else np.nan)
    return out


def main():
    rng = np.random.default_rng(SEED)
    X, sig = load()
    fits = fit_signatures(X, sig, rng)
    print(f"{len(fits)} signatures with >= {MIN_EDGES} edges "
          f"(of {len(sig)} total)\n")

    print("SIGNATURES, AND HOW RELIABLE EACH ONE'S MAP IS\n")
    print(f"{'signature':<44}{'edges':>7}{'noise floor':>13}  family")
    print("  noise floor = median rotation between maps fitted on two halves of the same edges;")
    print("  two maps cannot be called different unless they exceed it")
    for key, (O, m, floor) in sorted(fits.items(), key=lambda kv: -kv[1][1]):
        st, k, tt = key
        print(f"{st + ' -' + k + '-> ' + tt:<44}{m:>7}{floor:>12.1f}°  {family(k)}")

    # ---- the floor turned out to measure something other than what it was for
    print("\n\nQ0 — WHAT THE NOISE FLOOR ACTUALLY MEASURES\n")
    print("  it was built to control for sample size. It does not track sample size:")
    print("  993 edges gives 68.7 deg, 97 edges gives 13.1. So it is not measuring")
    print("  sampling error -- it is measuring whether a consistent rotation EXISTS.")
    print("  Two halves of the same relation agreeing to 5 deg means the relation has")
    print("  one map; two halves landing 85 deg apart (random in O(8) is ~90) means")
    print("  the fitted map is an artifact of which edges happened to be drawn.\n")
    print(f"{'signature':<44}{'edges':>7}{'floor':>8}  reliability")
    order = sorted(fits.items(), key=lambda kv: (kv[1][2] if kv[1][2] > 1e-6 else 1e3))
    reliable = set()
    for key, (O, m, floor) in order:
        st, k, tt = key
        if floor < 1e-6:
            tag = "DEGENERATE — target cloud is a point, map is meaningless"
        elif floor < 25:
            tag = "reliable"
            reliable.add(key)
        elif floor < 45:
            tag = "weak"
        else:
            tag = "NO CONSISTENT ROTATION"
        print(f"{st + ' -' + k + '-> ' + tt:<44}{m:>7}{floor:>7.1f}°  {tag}")
    print(f"\n  reliable signatures: {len(reliable)} of {len(fits)}")
    rel_rels = sorted({k[1] for k in reliable})
    print(f"  relations surviving: {', '.join(rel_rels)}")
    print(f"  relations with NO reliable signature: "
          f"{', '.join(sorted({k[1] for k in fits} - set(rel_rels)))}")

    # ---- Q1: does a relation agree with itself across its own signatures?
    print("\n\nQ1 — DOES EACH RELATION AGREE WITH ITSELF ACROSS ITS OWN SIGNATURES?\n")
    print("  if a relation's own signatures are further apart than their noise floors,")
    print("  the pooled single-map fit of 6b was averaging incompatible things\n")
    by_rel = collections.defaultdict(list)
    for key in fits:
        by_rel[key[1]].append(key)

    self_consistent = {}
    print(f"{'relation':<20}{'sigs':>6}{'median gap':>13}{'median floor':>15}{'ratio':>8}  verdict")
    for k, keys in sorted(by_rel.items(), key=lambda kv: -len(kv[1])):
        if len(keys) < 2:
            continue
        gaps, floors = [], []
        for a, b in itertools.combinations(keys, 2):
            gaps.append(rotation(fits[a][0], fits[b][0]))
            floors.append(max(fits[a][2], fits[b][2]))
        g, f = float(np.median(gaps)), float(np.median(floors))
        ratio = g / f if f > 1e-9 else np.nan
        # "the gap is within the floor" only means agreement when the floor is
        # itself small. A relation whose halves land 85 deg apart has no map to
        # agree with, so a gap of 78 deg is not coherence -- it is noise on both
        # sides, and calling it coherent would be the whole trap of this section.
        if f > 45:
            verdict = "vacuous — no map to agree with (floor is noise)"
        elif ratio < 1.5:
            verdict = "coherent"
        else:
            verdict = "INCOHERENT — pooling was averaging incompatible maps"
        self_consistent[k] = verdict
        print(f"{k:<20}{len(keys):>6}{g:>12.1f}°{f:>14.1f}°{ratio:>8.2f}  {verdict}")

    # ---- Q2: is there still a two-family split at the signature level?
    print("\n\nQ2 — IS THERE STILL A TWO-FAMILY SPLIT, SIGNATURE BY SIGNATURE?\n")
    keys = sorted(fits)
    lab = [family(k[1]) for k in keys]
    within_isa, within_uses, across = [], [], []
    for i, j in itertools.combinations(range(len(keys)), 2):
        r = rotation(fits[keys[i]][0], fits[keys[j]][0])
        fl = max(fits[keys[i]][2], fits[keys[j]][2])
        rec = (r, fl, r / fl if fl > 1e-9 else np.nan)
        if lab[i] == lab[j] == "is-a":
            within_isa.append(rec)
        elif lab[i] == lab[j] == "uses":
            within_uses.append(rec)
        elif {lab[i], lab[j]} == {"is-a", "uses"}:
            across.append(rec)

    print(f"{'comparison':<28}{'n':>6}{'median rot':>13}{'median floor':>15}{'ratio':>8}")
    for name, rows in (("within is-a", within_isa),
                       ("within uses", within_uses),
                       ("is-a vs uses", across)):
        if rows:
            r = float(np.median([x[0] for x in rows]))
            f = float(np.median([x[1] for x in rows]))
            print(f"{name:<28}{len(rows):>6}{r:>12.1f}°{f:>14.1f}°{r / f:>8.2f}")

    if within_isa and within_uses and across:
        wi = np.median([x[0] for x in within_isa])
        wu = np.median([x[0] for x in within_uses])
        ac = np.median([x[0] for x in across])
        print(f"\n  a real two-family split needs  across > both withins.")
        print(f"  across {ac:.1f}°  vs  within-is-a {wi:.1f}°  and  within-uses {wu:.1f}°")
        if ac > max(wi, wu) * 1.2:
            print("  => SPLIT SURVIVES typing the nodes")
        elif ac < max(wi, wu):
            print("  => SPLIT DOES NOT SURVIVE — the families are no more separated")
            print("     from each other than they are internally. F2 refuted.")
        else:
            print("  => INCONCLUSIVE — separation is within noise of the internal spread")

    print("\n\nQ2b — RESTRICTED TO SIGNATURES THAT HAVE A RELIABLE MAP\n")
    rk = sorted(reliable)
    if len(rk) < 2:
        print("  fewer than two reliable signatures; nothing to compare")
    else:
        fam = collections.Counter(family(k[1]) for k in rk)
        print(f"  {len(rk)} reliable signatures, by family: {dict(fam)}")
        for a, b in itertools.combinations(rk, 2):
            r = rotation(fits[a][0], fits[b][0])
            fl = max(fits[a][2], fits[b][2])
            mark = "  <- beyond both floors" if r > fl else ""
            print(f"    {a[1]}[{a[0]}->{a[2]}]  vs  {b[1]}[{b[0]}->{b[2]}]"
                  f"   {r:.0f}° (floor {fl:.0f}°){mark}")
        if fam.get("is-a", 0) == 0:
            print("\n  NOT ONE subtyping signature has a reliable map. The is-a/uses")
            print("  comparison of 6b.3 was comparing a measured family against an")
            print("  unmeasurable one, and the 67-77 deg separation was the distance")
            print("  from real maps to noise. F2 is refuted for that reason, not")
            print("  because the architecture lacks the distinction.")

    # ---- where does ALGEBRA_VIOLATION actually sit?
    print("\n\nWHERE DOES ALGEBRA_VIOLATION SIT ONCE SIGNATURES ARE SEPARATED?\n")
    av = [k for k in fits if k[1] == "ALGEBRA_VIOLATION"]
    for a in av:
        rows = []
        for b in fits:
            if b == a:
                continue
            r = rotation(fits[a][0], fits[b][0])
            rows.append((r, b))
        rows.sort()
        st, k, tt = a
        print(f"  {st} -{k}-> {tt}  (floor {fits[a][2]:.1f}°)")
        print(f"    nearest : " + ", ".join(
            f"{b[1]}[{b[0]}->{b[2]}] {r:.0f}°" for r, b in rows[:3]))
        print(f"    furthest: " + ", ".join(
            f"{b[1]}[{b[0]}->{b[2]}] {r:.0f}°" for r, b in rows[-3:]))

    np.savez("typed_connection.npz",
             keys=np.array([f"{a}|{b}|{c}" for a, b, c in sorted(fits)]),
             maps=np.array([fits[k][0] for k in sorted(fits)]),
             counts=np.array([fits[k][1] for k in sorted(fits)]),
             floors=np.array([fits[k][2] for k in sorted(fits)]))
    print("\n  saved to typed_connection.npz")


if __name__ == "__main__":
    main()
