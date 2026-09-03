"""G7 — do we cluster LAYERS when a subsystem is a vertical SLICE?

Owner's argument, and it may be the most consequential thing raised in this arc.
An embedding places SIMILAR things together: all controllers near each other, all
DTOs near each other, all tests near each other. But a subsystem is not a set of
similar things. "Security" is a controller, a service, some config, some rules and
some DTOs — objects that do not resemble one another at all. So a subsystem is a
VERTICAL SLICE across layers, and similarity clustering finds HORIZONTAL layers.
A single subsystem naturally sits across many sub-topologies rather than being
bounded by one.

The circumstantial evidence is already in the subsystem names this pipeline
produced: dto, dto, dtos, service, service, exceptions, fixtures, showcases,
registry — layers — against subscription, consent, user, notification, faq —
features. Roughly half the partition is horizontal.

If the argument holds, four things must be true at once, and each is measurable
using entity_type as the layer coordinate (Actor / Process / Resource / Rule /
Context / Event is essentially controller / service / data / validation /
infrastructure / event):

  1. CO-CHANGE IS CROSS-LAYER. Files that change together should differ in entity
     type more often than chance, because a feature change touches its controller
     AND its service AND its DTO.

  2. CONTENT SIMILARITY IS WITHIN-LAYER. The embedding should rate same-type pairs
     as far more similar than cross-type pairs — it is reading the code, and a
     controller reads like a controller.

  3. TYPED EDGES ARE CROSS-LAYER. Actor -PERFORMS-> Process and
     Process -USES-> Resource cross layers by construction. This would explain the
     otherwise-odd result that typed edges carry a 14x conditional lift on 0.57%
     of pairs: they connect exactly the pairs the embedding cannot see, because
     those pairs are DISTANT in embedding space and adjacent in the architecture.

  4. THE DERIVED PARTITION IS LAYER-BIASED. Its clusters should be purer in
     entity type than a feature decomposition would be.

If all four hold, the arc's thin margins have a structural explanation: the
primary view was optimising for the wrong kind of proximity, and the typed graph
is not a weak supplement to it but the complementary half — which is also why
LINEAR blending could never work, since the two signals cover disjoint kinds of
pairs rather than the same pairs with different noise.
"""
import collections
import itertools
import subprocess

import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
                "RETURN id(n) AS id, n.file_path AS fp, n.entity_type AS et, "
                "n.embedding AS emb, n.v3_subsystem AS sub", ns=NS))
            edges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "RETURN id(a) AS s, id(b) AS t, type(r) AS k", ns=NS))
    finally:
        drv.close()

    n = len(nodes)
    idx = {r["id"]: i for i, r in enumerate(nodes)}
    et = np.array([r["et"] for r in nodes])
    sub = np.array([r["sub"] if r["sub"] is not None else -1 for r in nodes])
    repo_of, rel_of = {}, {}
    for i, r in enumerate(nodes):
        rest = (r["fp"] or "").replace("\\", "/")
        rest = rest[len(ROOT):] if rest.startswith(ROOT) else rest
        p = rest.split("/", 1)
        repo_of[i], rel_of[i] = (p[0], p[1]) if len(p) == 2 else ("?", rest)
    X = np.array([r["emb"] for r in nodes], dtype=np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    leaf = np.array([hash(repo_of[i] + "/" + rel_of[i].rsplit("/", 1)[0]) % 10 ** 9
                     for i in range(n)])

    print("entity-type distribution (the layer coordinate):")
    for t, c in collections.Counter(et).most_common():
        print(f"    {t:<12}{c:>6}")

    A = np.zeros((n, n), dtype=bool)
    for e in edges:
        u, v = idx.get(e["s"]), idx.get(e["t"])
        if u is not None and v is not None and u != v:
            A[u, v] = A[v, u] = True

    by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}
    lab = collections.Counter()
    for rp in REPOS:
        out = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        for b in out.split("\x01")[1:]:
            ids = sorted({by_repo[rp][f] for f in
                          {l.strip() for l in b.splitlines() if l.strip()}
                          if f in by_repo[rp]})
            if 2 <= len(ids) <= 30:
                for a_, b_ in itertools.combinations(ids, 2):
                    lab[(a_, b_)] += 1

    pairs = [(a, b) for a, b in itertools.combinations(range(n), 2)
             if repo_of[a] == repo_of[b]]
    P = np.array(pairs)
    y = np.array([1 if lab.get((a, b), 0) else 0 for a, b in pairs])
    same_type = et[P[:, 0]] == et[P[:, 1]]
    sim = np.einsum("ij,ij->i", X[P[:, 0]], X[P[:, 1]])
    has_edge = A[P[:, 0], P[:, 1]]

    print(f"\n\n{len(pairs):,} within-repo pairs; "
          f"{same_type.mean():.1%} are same-entity-type by chance\n")

    print("1. IS CO-CHANGE CROSS-LAYER?\n")
    print(f"{'pair kind':<22}{'pairs':>10}{'co-change rate':>16}{'lift':>8}")
    base = y.mean()
    for name, m in (("same entity type", same_type), ("cross entity type", ~same_type)):
        print(f"{name:<22}{int(m.sum()):>10,}{y[m].mean():>16.4f}{y[m].mean()/base:>8.2f}x")
    share = y[~same_type].sum() / y.sum()
    print(f"\n  share of ALL co-change that is cross-type: {share:.1%}")
    print(f"  share expected if coupling ignored type    : {(~same_type).mean():.1%}")

    print("\n\n2. IS CONTENT SIMILARITY WITHIN-LAYER?\n")
    print(f"  mean cosine, same type  : {sim[same_type].mean():.4f}")
    print(f"  mean cosine, cross type : {sim[~same_type].mean():.4f}")
    print(f"  gap                     : {sim[same_type].mean() - sim[~same_type].mean():+.4f}")
    top = np.argsort(-sim)[:int(0.01 * len(sim))]
    print(f"  among the top 1% most similar pairs, "
          f"{same_type[top].mean():.1%} are same type (chance {same_type.mean():.1%})")

    print("\n\n3. ARE TYPED EDGES CROSS-LAYER?\n")
    print(f"  edges among same-type pairs : {int(has_edge[same_type].sum()):,}")
    print(f"  edges among cross-type pairs: {int(has_edge[~same_type].sum()):,}")
    ecross = has_edge[~same_type].sum() / max(has_edge.sum(), 1)
    print(f"  share of edges that cross a layer: {ecross:.1%} "
          f"(chance {(~same_type).mean():.1%})")

    print("\n\n4. WHERE DOES EACH SIGNAL FIND ITS CO-CHANGE?\n")
    print(f"{'signal':<34}{'cross-type share of its hits':>30}")
    hi_sim = sim >= np.quantile(sim, 0.99)
    for name, m in (("content embedding, top 1% pairs", hi_sim),
                    ("typed edge present", has_edge),
                    ("actual co-change", y.astype(bool))):
        if m.sum():
            print(f"{name:<34}{(~same_type)[m].mean():>29.1%}")
    print("\n  if co-change is cross-type while the embedding's hits are same-type,")
    print("  the primary view is optimising for the wrong kind of proximity")

    print("\n\n5. IS THE DERIVED PARTITION LAYER-BIASED?\n")

    def type_entropy(labels):
        """Mean within-cluster entropy of entity type, weighted by cluster size.
        High = mixed layers = feature-like. Low = pure = layer-like."""
        tot, wsum = 0.0, 0
        for c in set(labels):
            m = labels == c
            cnt = collections.Counter(et[m])
            k = m.sum()
            h = -sum((v / k) * np.log2(v / k) for v in cnt.values() if v)
            tot += h * k
            wsum += k
        return tot / max(wsum, 1)

    glob = collections.Counter(et)
    hglob = -sum((v / n) * np.log2(v / n) for v in glob.values())
    print(f"  global entity-type entropy (the ceiling): {hglob:.3f} bits")
    for name, l in (("derived subsystems", sub), ("directory leaf", leaf)):
        h = type_entropy(l)
        print(f"  {name:<24}{h:>8.3f} bits  ({h/hglob:.0%} of ceiling)")
    print("\n  a feature decomposition keeps its clusters mixed and sits near the")
    print("  ceiling; a layer decomposition purifies them and sits well below it")


if __name__ == "__main__":
    main()
