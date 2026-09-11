"""
Does the shared content signal swamp the relation-specific one?

Section 6 of the V3 foundations paper proposes a mechanism: at propertyRatio 0.5
every per-relation projection inherits the same dominant content signal, which is
why the projections are mutually aligned at 0.67-0.95, and the architecture lives
only in the smaller adjacency-driven perturbation.

If that is right, the effect should depend on propertyRatio in a specific way:

  ratio -> 1.0   all content, no structure  -> corrections vanish, no signal
  ratio -> 0.0   no content, pure structure -> the V2 regime, random projection
                                               on sparse adjacency, sign noise

so the signed structure should be strongest somewhere in between, and the
label-permutation null should track it.

Reports, for each ratio: the observed correction cosine for MODIFIES vs CALLS,
the mean over three label permutations, and the separation between them.
"""
import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS = "CheckItOutV3"
BEHAVIOURAL = ["PERFORMS", "USES", "MODIFIES", "CALLS", "ACCESSES"]
A_LABEL, B_LABEL = 2, 3          # MODIFIES, CALLS
RATIOS = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
PERMS = 3
GRAPH = "sweep"


def run(tx, q, **kw):
    return list(tx.run(q, **kw))


def tag_and_permute(driver, perms):
    with driver.session() as s:
        s.run(f"""
        MATCH (a:EntityDetail {{namespace:$ns}})-[r]->(b:EntityDetail {{namespace:$ns}})
        WHERE type(r) IN $rels
        SET r.sw_true = CASE type(r)
              WHEN 'PERFORMS' THEN 0 WHEN 'USES' THEN 1 WHEN 'MODIFIES' THEN 2
              WHEN 'CALLS' THEN 3 ELSE 4 END
        """, ns=NS, rels=BEHAVIOURAL)
        s.run("""
        MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns})
        WHERE r.sw_true IS NOT NULL
        WITH collect(r) AS rels, collect(r.sw_true) AS labels
        UNWIND range(0, $p - 1) AS perm
        WITH rels, labels, perm,
             [x IN apoc.coll.sortMaps([y IN range(0, size(rels)-1) | {i:y, k:rand()}], 'k') | x.i] AS order
        UNWIND range(0, size(rels)-1) AS pos
        WITH rels[order[pos]] AS rel, labels[pos] AS lab, perm
        SET rel['sw_shuf_' + perm] = lab
        """, ns=NS, p=perms)


def project(driver, perms):
    props = "r { .sw_true, " + ", ".join(f".sw_shuf_{i}" for i in range(perms)) + " }"
    with driver.session() as s:
        s.run(f"""
        MATCH (source:EntityDetail {{namespace:$ns}})-[r]->(target:EntityDetail {{namespace:$ns}})
        WHERE r.sw_true IS NOT NULL
        WITH gds.graph.project('{GRAPH}', source, target, {{
            sourceNodeProperties: source {{ .embedding }},
            targetNodeProperties: target {{ .embedding }},
            relationshipProperties: {props}
        }}, {{ undirectedRelationshipTypes: ['*'] }}) AS g
        RETURN g.nodeCount AS n
        """, ns=NS)


def fastrp(driver, graph, ratio, prop, rel_filter=None):
    """Filter (optionally) then FastRP; return {nodeId: vector}."""
    name = graph
    with driver.session() as s:
        if rel_filter is not None:
            name = f"{graph}_f"
            s.run(f"CALL gds.graph.drop('{name}', false)")
            s.run(f"CALL gds.graph.filter('{name}', '{graph}', '*', $f)", f=rel_filter)
        cfg = {
            "embeddingDimension": 8,
            "featureProperties": ["embedding"],
            "propertyRatio": ratio,
            "iterationWeights": [0.0, 1.0, 1.0],
            "randomSeed": 42,
            "mutateProperty": prop,
        }
        if abs(ratio) < 1e-12:               # GDS rejects featureProperties at ratio 0
            cfg.pop("featureProperties"); cfg.pop("propertyRatio")
        s.run(f"CALL gds.fastRP.mutate('{name}', $c)", c=cfg)
        rows = list(s.run(
            f"CALL gds.graph.nodeProperty.stream('{name}', '{prop}') YIELD nodeId, propertyValue "
            f"RETURN nodeId, propertyValue"))
    return {r["nodeId"]: np.array(r["propertyValue"], dtype=np.float64) for r in rows}


def delta_cosine(base, A, B):
    vals = []
    for k, a in A.items():
        b, r0 = B.get(k), base.get(k)
        if b is None or r0 is None:
            continue
        if not np.abs(a).sum() or not np.abs(b).sum():
            continue
        da, db = a - r0, b - r0
        na, nb = np.linalg.norm(da), np.linalg.norm(db)
        if na == 0 or nb == 0:
            continue
        vals.append(float(da @ db / (na * nb)))
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def cleanup(driver, perms):
    with driver.session() as s:
        for g in (GRAPH, f"{GRAPH}_f"):
            s.run(f"CALL gds.graph.drop('{g}', false)")
        removes = ", ".join(["r.sw_true"] + [f"r.sw_shuf_{i}" for i in range(perms)])
        s.run(f"""
        MATCH (a:EntityDetail {{namespace:$ns}})-[r]->(b:EntityDetail {{namespace:$ns}})
        WHERE r.sw_true IS NOT NULL REMOVE {removes}
        """, ns=NS)


def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with driver.session() as s:
            s.run(f"CALL gds.graph.drop('{GRAPH}', false)")
        tag_and_permute(driver, PERMS)
        project(driver, PERMS)

        print(f"{'ratio':>6} {'observed':>10} {'n':>5} {'null mean':>10} {'separation':>11}")
        print("-" * 46)
        for ri, ratio in enumerate(RATIOS):
            # the base graph persists across iterations, so its mutate property
            # must be unique per ratio or GDS refuses the second write
            base = fastrp(driver, GRAPH, ratio, f"b0_{ri}")
            A = fastrp(driver, GRAPH, ratio, "pa", f"r.sw_true = {float(A_LABEL)}")
            B = fastrp(driver, GRAPH, ratio, "pb", f"r.sw_true = {float(B_LABEL)}")
            obs, n = delta_cosine(base, A, B)

            nulls = []
            for p in range(PERMS):
                An = fastrp(driver, GRAPH, ratio, f"na{p}", f"r.sw_shuf_{p} = {float(A_LABEL)}")
                Bn = fastrp(driver, GRAPH, ratio, f"nb{p}", f"r.sw_shuf_{p} = {float(B_LABEL)}")
                v, _ = delta_cosine(base, An, Bn)
                if not np.isnan(v):
                    nulls.append(v)
            nm = float(np.mean(nulls)) if nulls else float("nan")
            print(f"{ratio:>6.3f} {obs:>10.3f} {n:>5} {nm:>10.3f} {obs - nm:>11.3f}")
    finally:
        cleanup(driver, PERMS)
        driver.close()


if __name__ == "__main__":
    main()
