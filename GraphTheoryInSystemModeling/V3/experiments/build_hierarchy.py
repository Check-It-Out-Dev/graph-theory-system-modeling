"""S5 — build the 3-level hierarchy and write it back with its provenance.

The two measurable gates are met: 16.80x coherence against directory's 9.85x at
matched granularity on held-out commits, and bootstrap ARI 0.743. What remains is
to write the partition into the graph in a form that can be audited later and
updated incrementally.

Nesting, not three clusterings. A coarse Louvain pass gives subsystems at a
granularity a person can hold in their head; a second pass WITHIN each subsystem
gives modules. Running Louvain twice at different resolutions would give two
partitions that need not nest at all, and a hierarchy whose levels contradict each
other is worse than no hierarchy. Clustering inside each part guarantees the
containment.

Every subsystem node carries what produced it -- method, resolution, held-out
coherence, ARI, and the commit range the evaluation used -- so a later reader can
tell whether to trust it, and a delta run can tell what needs recomputing.
"""
import collections
import itertools
import subprocess

import numpy as np
import networkx as nx
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
KNN, SEED = 12, 42
RES_COARSE, RES_FINE = 1.0, 1.6


def main():
    rng = np.random.default_rng(SEED)
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.embedding IS NOT NULL "
                "RETURN id(n) AS id, n.name AS name, n.file_path AS fp, "
                "n.entity_type AS et, n.embedding AS emb", ns=NS))
    finally:
        drv.close()

    n = len(nodes)
    repo_of, rel_of = {}, {}
    for i, r in enumerate(nodes):
        rest = (r["fp"] or "").replace("\\", "/")
        rest = rest[len(ROOT):] if rest.startswith(ROOT) else rest
        p = rest.split("/", 1)
        repo_of[i], rel_of[i] = (p[0], p[1]) if len(p) == 2 else ("?", rest)
    X = np.array([r["emb"] for r in nodes], dtype=np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = X @ X.T
    np.fill_diagonal(S, -1.0)

    def graph_over(members):
        g = nx.Graph()
        g.add_nodes_from(members)
        ms = set(members)
        for i in members:
            for j in np.argsort(-S[i])[:60]:
                j = int(j)
                if j in ms and repo_of[i] == repo_of[j] and S[i, j] > 0:
                    g.add_edge(int(i), j, weight=float(S[i, j]))
                    if g.degree[int(i)] >= KNN:
                        break
        return g

    def louvain(g, res):
        return nx.community.louvain_communities(g, weight="weight", resolution=res, seed=SEED)

    coarse = louvain(graph_over(list(range(n))), RES_COARSE)
    coarse = sorted(coarse, key=len, reverse=True)
    sub_of = np.full(n, -1)
    for c, grp in enumerate(coarse):
        for v in grp:
            sub_of[v] = c

    mod_of = np.full(n, -1)
    next_mod = 0
    for c, grp in enumerate(coarse):
        members = sorted(grp)
        if len(members) < 8:
            for v in members:
                mod_of[v] = next_mod
            next_mod += 1
            continue
        for sub in louvain(graph_over(members), RES_FINE):
            for v in sub:
                mod_of[v] = next_mod
            next_mod += 1

    print(f"level 2: {len(coarse)} subsystems")
    print(f"level 3: {next_mod} modules, nested inside them by construction\n")

    # held-out coherence for both levels
    by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}
    te = collections.Counter()
    span = []
    for rp in REPOS:
        out = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        sets = [sorted({l.strip() for l in b.splitlines() if l.strip()})
                for b in out.split("\x01")[1:]]
        sets = [s for s in sets if 2 <= len(s) <= 30]
        span.append(f"{rp}:{len(sets)}")
        m = rng.random(len(sets)) < 0.5
        for files, keep in zip(sets, m):
            if keep:
                continue
            ids = sorted({by_repo[rp][f] for f in files if f in by_repo[rp]})
            for a, b in itertools.combinations(ids, 2):
                te[(a, b)] += 1
    pairs = [(a, b) for a, b in itertools.combinations(range(n), 2)
             if repo_of[a] == repo_of[b]]
    P = np.array(pairs)
    y = np.array([1 if te.get((a, b), 0) else 0 for a, b in pairs])

    def coh(part):
        same = part[P[:, 0]] == part[P[:, 1]]
        return float(y[same].mean() / max(y[~same].mean(), 1e-12))

    coh_sub, coh_mod = coh(sub_of), coh(mod_of)
    print(f"held-out coherence: subsystems {coh_sub:.2f}x, modules {coh_mod:.2f}x\n")

    # name each subsystem by its most common directory, for readability
    names = {}
    for c, grp in enumerate(coarse):
        d = collections.Counter(rel_of[v].rsplit("/", 1)[0].split("/")[-1] for v in grp)
        et = collections.Counter(nodes[v]["et"] for v in grp)
        names[c] = (d.most_common(1)[0][0] if d else f"sub{c}", et.most_common(1)[0][0])

    print(f"{'#':>3}{'files':>7}  {'dominant folder':<28}{'dominant type':<12}")
    for c, grp in enumerate(coarse[:12]):
        print(f"{c:>3}{len(grp):>7}  {names[c][0][:27]:<28}{names[c][1]:<12}")
    if len(coarse) > 12:
        print(f"     ... and {len(coarse) - 12} more")

    prov = {
        "method": "Louvain on kNN(k=12) cosine graph of Qwen3-Embedding-8B file embeddings, "
                  "within-repo edges only; modules by a second Louvain pass inside each subsystem",
        "res_coarse": RES_COARSE, "res_fine": RES_FINE,
        "coherence_subsystem": round(coh_sub, 3), "coherence_module": round(coh_mod, 3),
        "baseline_directory": 9.85, "baseline_incumbent": 1.36,
        "bootstrap_ari": 0.743, "eval": "held out by commit, 50/50 split; " + ", ".join(span),
        "typed_edges_contribution": "none measurable at partition level (0.57% pair coverage); "
                                    "typed edges give 14x conditional lift pairwise but do not "
                                    "improve clustering",
    }

    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            s.run("MATCH (m:V3Master {namespace:$ns}) DETACH DELETE m", ns=NS)
            s.run("MATCH (x:V3Subsystem {namespace:$ns}) DETACH DELETE x", ns=NS)
            s.run("""
            CREATE (m:V3Master {namespace:$ns, name:'CheckItOutV3 derived partition',
                   level:1, built:datetime(), method:$method,
                   res_coarse:$res_coarse, res_fine:$res_fine,
                   coherence_subsystem:$coherence_subsystem, coherence_module:$coherence_module,
                   baseline_directory:$baseline_directory, baseline_incumbent:$baseline_incumbent,
                   bootstrap_ari:$bootstrap_ari, eval:$eval,
                   typed_edges_contribution:$typed_edges_contribution,
                   subsystem_count:$nsub, module_count:$nmod, file_count:$nfile})
            """, ns=NS, nsub=len(coarse), nmod=next_mod, nfile=n, **prov)

            rows = [{"c": c, "name": names[c][0], "et": names[c][1],
                     "size": len(grp),
                     "repos": sorted({repo_of[v] for v in grp})} for c, grp in enumerate(coarse)]
            s.run("""
            UNWIND $rows AS r
            MATCH (m:V3Master {namespace:$ns})
            CREATE (x:V3Subsystem {namespace:$ns, level:2, subsystem:r.c,
                    name:r.name, dominant_type:r.et, size:r.size, repos:r.repos})
            CREATE (m)-[:HAS_SUBSYSTEM]->(x)
            """, ns=NS, rows=rows)

            assign = [{"id": nodes[i]["id"], "s": int(sub_of[i]), "m": int(mod_of[i])}
                      for i in range(n)]
            s.run("""
            UNWIND $rows AS r
            MATCH (n) WHERE id(n) = r.id
            MATCH (x:V3Subsystem {namespace:$ns, subsystem:r.s})
            SET n.v3_subsystem = r.s, n.v3_module = r.m
            MERGE (x)-[:CONTAINS]->(n)
            """, ns=NS, rows=assign)
    finally:
        drv.close()

    print(f"\nwritten: 1 V3Master -> {len(coarse)} V3Subsystem -> {n} files, "
          f"with v3_module on each file for the level-3 refinement")


if __name__ == "__main__":
    main()
