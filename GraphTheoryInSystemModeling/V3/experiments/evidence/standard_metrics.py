"""G4b — the field-standard metrics, so results are comparable with published work.

Everything measured so far used objectives invented for this arc: "coherence",
plus modularity of a co-change graph. Both are defensible, neither is what the
software-architecture-recovery literature reports. That makes these results hard
to place against published numbers, and it is easy to fix.

Three standards, computed here for every partition:

  TurboMQ   Modularization Quality (Mancoridis et al.). Intrinsic, needs no
            ground truth. CF_i = 2*mu_i / (2*mu_i + sum_j eps_ij) per cluster,
            summed. Rewards internal edges, penalises crossing ones. This is the
            number most module-clustering papers optimise.

  MoJo      Minimum Move+Join operations to transform one partition into another
            (Tzerpos & Holt). Reported both as a raw distance and as MoJoFM, the
            normalised percentage. The reference partition here is DIRECTORY
            STRUCTURE, on the argument that a directory tree IS a
            developer-created decomposition -- the same argument the literature
            makes when it uses an expert decomposition as ground truth.

  ARI/NMI   Unambiguous agreement measures, reported alongside because MoJoFM's
            normalisation convention varies between implementations and I would
            rather not rest a comparison on a convention I cannot verify.

Interpretation stated in advance so it is not chosen after seeing the numbers:
high MoJoFM against directory means the method recovered what developers already
encoded in the folder tree -- reassuring, but it also means the method added
little. LOW agreement is only a success if it is paid for by beating directory on
the held-out co-change metrics. Neither direction is automatically good.
"""
import collections
import itertools

import numpy as np
import networkx as nx
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"


def turbo_mq(labels, G):
    """Sum over clusters of CF_i = 2*mu_i / (2*mu_i + sum of crossing edges)."""
    intra = collections.Counter()
    inter = collections.Counter()
    for u, v in G.edges():
        a, b = labels[u], labels[v]
        if a == b:
            intra[a] += 1
        else:
            inter[a] += 1
            inter[b] += 1
    total = 0.0
    for c in set(labels):
        mu, eps = intra[c], inter[c]
        denom = 2 * mu + eps
        if denom > 0:
            total += 2 * mu / denom
    return float(total), len(set(labels))


def mojo(A, B):
    """Move+Join distance from partition A to partition B (Tzerpos & Holt).

    Each A-cluster is tagged with the B-cluster it overlaps most; objects not
    carrying their cluster's tag must MOVE, and clusters sharing a tag must JOIN.
    """
    n = len(A)
    byA = collections.defaultdict(list)
    for i, a in enumerate(A):
        byA[a].append(i)
    moves, tags = 0, []
    for a, members in byA.items():
        cnt = collections.Counter(B[i] for i in members)
        tag, best = cnt.most_common(1)[0]
        moves += len(members) - best
        tags.append(tag)
    joins = len(byA) - len(set(tags))
    return moves + joins


def mojofm(A, B):
    """Normalised: 100 * (1 - mno(A,B) / max_mno(B)).

    max_mno(B) is the distance from the worst possible partition, taken here as
    all-singletons, which needs n - |B| joins and no moves.
    """
    n = len(A)
    worst = n - len(set(B))
    if worst <= 0:
        return 100.0
    return 100.0 * (1.0 - mojo(A, B) / worst)


def ari_nmi(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    cm = collections.Counter(zip(a, b))
    ca, cb = collections.Counter(a), collections.Counter(b)

    def c2(x):
        return x * (x - 1) / 2

    sij = sum(c2(v) for v in cm.values())
    sa, sb = sum(c2(v) for v in ca.values()), sum(c2(v) for v in cb.values())
    N = c2(n)
    exp = sa * sb / N
    mx = (sa + sb) / 2
    ari = (sij - exp) / (mx - exp) if mx > exp else 1.0

    def H(c):
        return -sum((v / n) * np.log(v / n) for v in c.values() if v)

    ha, hb = H(ca), H(cb)
    mi = 0.0
    for (x, y), v in cm.items():
        mi += (v / n) * np.log((v / n) / ((ca[x] / n) * (cb[y] / n)))
    nmi = mi / max(np.sqrt(ha * hb), 1e-12)
    return float(ari), float(nmi)


def main():
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
                "RETURN id(n) AS id, n.file_path AS fp, n.v3_subsystem AS sub, "
                "n.v3_module AS mod, n.subsystem_id AS old", ns=NS))
            edges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "RETURN id(a) AS s, id(b) AS t", ns=NS))
    finally:
        drv.close()

    n = len(nodes)
    idx = {r["id"]: i for i, r in enumerate(nodes)}
    repo_of, rel_of = {}, {}
    for i, r in enumerate(nodes):
        rest = (r["fp"] or "").replace("\\", "/")
        rest = rest[len(ROOT):] if rest.startswith(ROOT) else rest
        p = rest.split("/", 1)
        repo_of[i], rel_of[i] = (p[0], p[1]) if len(p) == 2 else ("?", rest)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for e in edges:
        u, v = idx.get(e["s"]), idx.get(e["t"])
        if u is not None and v is not None and u != v:
            G.add_edge(u, v)

    sub = np.array([r["sub"] if r["sub"] is not None else -1 for r in nodes])
    mod = np.array([r["mod"] if r["mod"] is not None else -1 for r in nodes])
    old = np.array([r["old"] if r["old"] is not None else -1 for r in nodes])
    leaf = np.array([hash(repo_of[i] + "/" + rel_of[i].rsplit("/", 1)[0]) % 10 ** 9
                     for i in range(n)])
    d3 = np.array([hash(repo_of[i] + "/" +
                        "/".join(rel_of[i].rsplit("/", 1)[0].split("/")[:3])) % 10 ** 9
                   for i in range(n)])
    single = np.zeros(n, dtype=int)
    singles = np.arange(n)

    print(f"{n} files, {G.number_of_edges()} deduplicated structural edges\n")
    print("TurboMQ — intrinsic, on the typed dependency graph\n")
    print(f"{'partition':<28}{'parts':>7}{'TurboMQ':>10}{'MQ/part':>10}")
    parts = {
        "derived subsystems": sub,
        "derived modules": mod,
        "directory leaf": leaf,
        "directory depth 3": d3,
        "incumbent subsystem_id": old,
        "— single part (sanity)": single,
        "— all singletons (sanity)": singles,
    }
    for name, lab in parts.items():
        mq, k = turbo_mq(lab, G)
        print(f"{name:<28}{k:>7}{mq:>10.2f}{mq / max(k, 1):>10.4f}")
    print("\n  TurboMQ rises with part count by construction (each cluster adds at")
    print("  most 1.0), so the per-part column is the comparable one. Singletons")
    print("  score 0 because a cluster with no internal edge contributes nothing.")

    print("\n\nAGREEMENT WITH THE DEVELOPER DECOMPOSITION (directory leaf)\n")
    print(f"{'partition':<28}{'MoJo':>8}{'MoJoFM':>9}{'ARI':>8}{'NMI':>8}")
    for name, lab in (("derived subsystems", sub), ("derived modules", mod),
                      ("directory depth 3", d3), ("incumbent subsystem_id", old)):
        m = mojo(lab, leaf)
        fm = mojofm(lab, leaf)
        a, nm = ari_nmi(lab, leaf)
        print(f"{name:<28}{m:>8}{fm:>8.1f}%{a:>8.3f}{nm:>8.3f}")

    print("\n  Read with the interpretation fixed in advance: high agreement means")
    print("  the method recovered what the folder tree already encodes, which is")
    print("  reassuring but adds little. Low agreement is a success only if paid")
    print("  for by beating directory on held-out co-change — which the derived")
    print("  subsystems do at directory's own granularity (+0.165 modularity,")
    print("  20/20 splits) and only marginally at matched granularity")
    print("  (+0.038 ± 0.033, 17/20).")

    print("\n\nSANITY: do the degenerate partitions score as they must?\n")
    mq1, _ = turbo_mq(single, G)
    mqs, _ = turbo_mq(singles, G)
    print(f"  single part   TurboMQ {mq1:.2f}  (one cluster, no crossing edges -> 1.00)")
    print(f"  all singletons TurboMQ {mqs:.2f}  (no internal edges anywhere -> 0.00)")
    ok = abs(mq1 - 1.0) < 1e-6 and abs(mqs) < 1e-6
    print(f"  behaves as specified: {ok}")


if __name__ == "__main__":
    main()
