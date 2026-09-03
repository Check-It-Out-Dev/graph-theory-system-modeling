"""S4 — hyperedges, subsystem density, and the typed graph that survives collapsing.

A pair is a poor model of a change. A commit touches a SET of files, and the set
is the unit of work; reducing it to pairs throws away the arity and double-counts
big commits. So take hyperedges seriously and ask three things.

  1. CONTAINMENT. Does the derived partition hold commits together? A good
     subsystem boundary is one that changes rarely cross. Measured as the
     fraction of commit-hyperedges lying entirely inside one subsystem, and the
     mean number of subsystems a commit spans.

     The null matters more than the statistic. Containment rises automatically
     when parts are large -- a partition with one giant part contains everything
     -- so the comparison is against LABEL PERMUTATIONS, which preserve the part
     size distribution exactly and destroy only the assignment. 200 of them.

  2. COLLAPSING. Once a partition exists, an edge inside a subsystem and an edge
     between two subsystems play different roles. Intra edges are density: they
     describe how tightly a subsystem is wired internally. Inter edges are the
     level-2 structure: collapsing the graph onto subsystems leaves a small typed
     multigraph whose arrows are the architectural seams. That collapsed object
     is the "typed sub-topology" at subsystem level, and it is small enough to
     read.

  3. MODEL HYPEREDGES. The 6-entity model gives a second construction that owes
     nothing to git: a Process together with the Actors that invoke it and the
     Resources it touches is one behavioural unit. If those units also respect
     the partition, the partition is capturing behaviour and not just vocabulary,
     since the embedding that produced it never saw the edges.
"""
import collections
import itertools
import subprocess

import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]
MAX_COMMIT_FILES, N_PERM, SEED = 30, 200, 42


def load():
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes = list(s.run(
                "MATCH (n:EntityDetail {namespace:$ns}) WHERE n.v3_subsystem IS NOT NULL "
                "RETURN id(n) AS id, n.name AS name, n.file_path AS fp, n.entity_type AS et, "
                "n.v3_subsystem AS sub, n.v3_module AS mod", ns=NS))
            edges = list(s.run(
                "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
                "RETURN id(a) AS s, id(b) AS t, type(r) AS k", ns=NS))
            subs = list(s.run(
                "MATCH (x:V3Subsystem {namespace:$ns}) "
                "RETURN x.subsystem AS id, x.name AS name, x.size AS size "
                "ORDER BY x.subsystem", ns=NS))
    finally:
        drv.close()
    return nodes, edges, subs


def main():
    rng = np.random.default_rng(SEED)
    nodes, edges, subs = load()
    n = len(nodes)
    idx = {r["id"]: i for i, r in enumerate(nodes)}
    sub = np.array([r["sub"] for r in nodes])
    name_of = {r["id"]: r["name"] for r in subs}
    repo_of, rel_of = {}, {}
    for i, r in enumerate(nodes):
        rest = (r["fp"] or "").replace("\\", "/")
        rest = rest[len(ROOT):] if rest.startswith(ROOT) else rest
        p = rest.split("/", 1)
        repo_of[i], rel_of[i] = (p[0], p[1]) if len(p) == 2 else ("?", rest)
    dirlab = np.array([hash(repo_of[i] + "/" + rel_of[i].rsplit("/", 1)[0]) % 10 ** 9
                       for i in range(n)])
    print(f"{n} files, {len(set(sub))} subsystems, {len(edges)} typed edges\n")

    # ---------------------------------------------------------------- 1. commits
    by_repo = {rp: {rel_of[i]: i for i in range(n) if repo_of[i] == rp} for rp in REPOS}
    hyper = []
    for rp in REPOS:
        out = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        for block in out.split("\x01")[1:]:
            files = {l.strip() for l in block.splitlines() if l.strip()}
            ids = sorted({by_repo[rp][f] for f in files if f in by_repo[rp]})
            if 2 <= len(ids) <= MAX_COMMIT_FILES:
                hyper.append(ids)
    print(f"1. COMMIT HYPEREDGES — {len(hyper)} commits touching 2..{MAX_COMMIT_FILES} "
          f"known files\n")

    def contain(lab):
        pure = sum(1 for h in hyper if len({lab[i] for i in h}) == 1)
        span = float(np.mean([len({lab[i] for i in h}) for h in hyper]))
        return pure / len(hyper), span

    p_sub, s_sub = contain(sub)
    p_dir, s_dir = contain(dirlab)

    nulls = []
    for _ in range(N_PERM):
        nulls.append(contain(sub[rng.permutation(n)])[0])
    nm, nsd = float(np.mean(nulls)), float(np.std(nulls))
    z = (p_sub - nm) / nsd if nsd > 1e-12 else np.nan

    print(f"{'partition':<28}{'commits held whole':>20}{'subsystems spanned':>20}")
    print(f"{'derived subsystems (18)':<28}{p_sub:>19.1%}{s_sub:>20.2f}")
    print(f"{'leaf directory (252)':<28}{p_dir:>19.1%}{s_dir:>20.2f}")
    print(f"{'label-permutation null':<28}{nm:>19.1%}{'':<20}")
    print(f"\n  null preserves the part sizes exactly and destroys only the assignment,")
    print(f"  which is what stops a big-part partition from winning by construction.")
    print(f"  z = {z:+.1f} over {N_PERM} permutations "
          f"(null sd {nsd:.4f})")
    print(f"  NB the directory partition has 252 parts against 18, so its lower")
    print(f"  containment is expected -- finer partitions cut more commits. Not a")
    print(f"  like-for-like comparison, and reported only for scale.")

    # ---------------------------------------------------------------- 2. collapse
    print("\n\n2. COLLAPSING — what survives when each subsystem becomes one node\n")
    intra = collections.Counter()
    inter = collections.Counter()
    inter_typed = collections.Counter()
    for e in edges:
        u, v = idx.get(e["s"]), idx.get(e["t"])
        if u is None or v is None or u == v:
            continue
        a, b = sub[u], sub[v]
        if a == b:
            intra[a] += 1
        else:
            inter[(min(a, b), max(a, b))] += 1
            inter_typed[(min(a, b), max(a, b), e["k"])] += 1
    tot = sum(intra.values()) + sum(inter.values())
    print(f"  {sum(intra.values()):,} of {tot:,} typed edges are INTRA-subsystem "
          f"({sum(intra.values()) / tot:.1%})")
    print(f"  {sum(inter.values()):,} are INTER, spread over {len(inter)} of the "
          f"{len(subs) * (len(subs) - 1) // 2} possible subsystem pairs\n")

    sizes = {r["id"]: r["size"] for r in subs}
    print(f"{'#':>3}{'files':>7}{'intra edges':>13}{'density':>10}{'ext ratio':>11}  name")
    print("  density = intra edges / possible pairs; ext ratio = inter / (intra+inter)")
    for sid in sorted(sizes):
        sz = sizes[sid]
        poss = sz * (sz - 1) / 2
        ext = sum(c for (a, b), c in inter.items() if a == sid or b == sid)
        dens = intra[sid] / poss if poss else 0.0
        ratio = ext / max(intra[sid] + ext, 1)
        print(f"{sid:>3}{sz:>7}{intra[sid]:>13}{dens:>10.4f}{ratio:>11.2f}  {name_of[sid]}")

    print("\n  the strongest seams — subsystem pairs with the most crossing edges:\n")
    for (a, b), c in inter.most_common(10):
        types = collections.Counter({k: v for (x, y, k), v in inter_typed.items()
                                     if (x, y) == (a, b)})
        top = ", ".join(f"{k} {v}" for k, v in types.most_common(3))
        print(f"    {name_of[a]:>22} — {name_of[b]:<22} {c:>4} edges   [{top}]")

    # ---------------------------------------------------------------- 3. model
    print("\n\n3. MODEL HYPEREDGES — a Process with the Actors that invoke it and the\n"
          "   Resources it touches, one behavioural unit, owing nothing to git\n")
    nb = collections.defaultdict(set)
    for e in edges:
        u, v = idx.get(e["s"]), idx.get(e["t"])
        if u is None or v is None or u == v:
            continue
        if nodes[v]["et"] == "Process" and e["k"] in ("PERFORMS", "CALLS"):
            nb[v].add(u)
        if nodes[u]["et"] == "Process" and e["k"] in ("USES", "MODIFIES", "ACCESSES"):
            nb[u].add(v)
    units = [sorted({p} | s) for p, s in nb.items() if len(s) >= 1 and len({p} | s) >= 2]
    if units:
        pure = sum(1 for h in units if len({sub[i] for i in h}) == 1) / len(units)
        span = float(np.mean([len({sub[i] for i in h}) for h in units]))
        mn = []
        for _ in range(N_PERM):
            pm = sub[rng.permutation(n)]
            mn.append(sum(1 for h in units if len({pm[i] for i in h}) == 1) / len(units))
        mnm, mnsd = float(np.mean(mn)), float(np.std(mn))
        mz = (pure - mnm) / mnsd if mnsd > 1e-12 else np.nan
        print(f"  {len(units)} behavioural units, mean arity "
              f"{np.mean([len(h) for h in units]):.1f}")
        print(f"  held whole by the partition : {pure:.1%}")
        print(f"  label-permutation null      : {mnm:.1%}   z = {mz:+.1f}")
        print(f"  subsystems spanned, mean    : {span:.2f}")
        print("\n  the partition came from file CONTENT and never saw these edges, so")
        print("  containment above the null means it recovered behavioural units it")
        print("  was not shown.")
    else:
        print("  no units found")


if __name__ == "__main__":
    main()
