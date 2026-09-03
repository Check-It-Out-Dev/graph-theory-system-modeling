"""The single-writer global hyperedge emit — task #41, the H3 survivor.

Three shard-parallel attempts at this produced three membership semantics and two
key schemes (F83/F89/F94), so the design law is now: shared n-ary layers get ONE
writer, and this is it. Run whole-graph, no shard predicate, idempotent via MERGE
on a hub-keyed identity that is stable under incremental reindex.

Convention (frozen in HypatiaV5 §6):

  :HyperedgeCandidate {namespace, key, metapath, arity, member_ids, member_names,
                       hub_id, hub_name, via_relations, idf_weight,
                       precision_prior, source:'metapath-v3'}
  (n)-[:IN_HYPEREDGE {role:'hub'|'member', via}]->(h)
  key = f"{metapath}:hub:{hub_neo4j_id}"

The three meta-paths and their measured pair-precision priors (F79):

  P_R_P  hub=Resource,  satellites=Processes via USES|MODIFIES   prior 0.491
  A_P_A  hub=Process,   satellites=Actors via PERFORMS           prior 0.374
  A_P_R  hub=Process,   members=Actors (PERFORMS) + Resources
         (USES|MODIFIES) — the vertical slice                    prior 0.306

IDF centre weighting (F86): idf_weight = max(0, ln(N_hubs / k_satellites)) per
metapath. RepositoryResolver-class infrastructure hubs (k=15 minting 105
near-zero-evidence pairs) get weights near zero instead of dominating the layer.

--validate computes the number that justifies the weighting: within-hyperedge
pair precision against git co-change, unweighted vs IDF-weighted. If weighting
does not raise precision, F86 was wrong and the flag says so.
"""
import argparse
import collections
import itertools
import math
import subprocess

import numpy as np
from neo4j import GraphDatabase

URI, AUTH = "bolt://127.0.0.1:7611", ("neo4j", "password")
NS, ROOT = "CheckItOutV3", "C:/Users/Norbert/IdeaProjects/"
REPOS = ["checkItOut-be2", "checkItOut-fe-greenfield"]

SPECS = [
    ("P_R_P", "Resource", ["USES", "MODIFIES"], "in", 0.491),
    ("A_P_A", "Process", ["PERFORMS"], "in", 0.374),
]
# A_P_R is built from A_P_A's actors plus the hub's outgoing resources.
APR_PRIOR = 0.306


def load(session):
    nodes = {r["id"]: (r["name"], r["et"], r["fp"]) for r in session.run(
        "MATCH (n:EntityDetail {namespace:$ns}) "
        "RETURN id(n) AS id, n.name AS name, n.entity_type AS et, n.file_path AS fp",
        ns=NS)}
    edges = list(session.run(
        "MATCH (a:EntityDetail {namespace:$ns})-[r]->(b:EntityDetail {namespace:$ns}) "
        "WHERE type(r) IN ['USES','MODIFIES','PERFORMS'] "
        "RETURN id(a) AS s, id(b) AS t, type(r) AS k", ns=NS))
    return nodes, edges


def build(nodes, edges):
    into = collections.defaultdict(lambda: collections.defaultdict(set))   # hub -> rel -> srcs
    outof = collections.defaultdict(lambda: collections.defaultdict(set))  # src -> rel -> tgts
    for e in edges:
        into[e["t"]][e["k"]].add(e["s"])
        outof[e["s"]][e["k"]].add(e["t"])

    hyper = []
    # P_R_P and A_P_A — cohort of satellites pointing INTO the hub
    for mp, hub_type, rels, _, prior in SPECS:
        for hub, by_rel in into.items():
            if nodes[hub][1] != hub_type:
                continue
            sats = {}
            for rel in rels:
                for s in by_rel.get(rel, ()):
                    sats.setdefault(s, rel)
            if len(sats) < 2:
                continue
            hyper.append((mp, hub, dict(sats), prior))
    # A_P_R — hub Process, actors in, resources out; arity >= 3
    for hub, by_rel in into.items():
        if nodes[hub][1] != "Process":
            continue
        actors = {a: "PERFORMS" for a in by_rel.get("PERFORMS", ())}
        resources = {}
        for rel in ("USES", "MODIFIES"):
            for t in outof.get(hub, {}).get(rel, ()):
                resources.setdefault(t, rel)
        if actors and resources:
            hyper.append(("A_P_R", hub, {**actors, **resources}, APR_PRIOR))

    # IDF per metapath
    by_mp = collections.defaultdict(list)
    for mp, hub, sats, prior in hyper:
        by_mp[mp].append((hub, sats, prior))
    rows = []
    for mp, items in by_mp.items():
        N = len(items)
        for hub, sats, prior in items:
            idf = max(0.0, math.log(N / max(len(sats), 1)))
            rows.append({
                "key": f"{mp}:hub:{hub}", "metapath": mp, "hub_id": hub,
                "hub_name": nodes[hub][0], "arity": len(sats) + 1,
                "member_ids": sorted(sats), "via": sats,
                "member_names": [nodes[m][0] for m in sorted(sats)],
                "via_relations": sorted(set(sats.values())),
                "idf_weight": round(idf, 4), "precision_prior": prior,
            })
    return rows


def emit(session, rows):
    payload = [{**r, "via": None,
                "links": [{"id": m, "via": v} for m, v in r["via"].items()]}
               for r in rows]
    session.run("""
    UNWIND $rows AS r
    MERGE (h:HyperedgeCandidate {namespace:$ns, key:r.key})
    SET h.metapath=r.metapath, h.arity=r.arity, h.member_ids=r.member_ids,
        h.member_names=r.member_names, h.hub_id=r.hub_id, h.hub_name=r.hub_name,
        h.via_relations=r.via_relations, h.idf_weight=r.idf_weight,
        h.precision_prior=r.precision_prior, h.source='metapath-v3',
        h.idf_formula='max(0, ln(N_hubs/k_satellites))'
    WITH h, r
    MATCH (hub) WHERE id(hub) = r.hub_id
    MERGE (hub)-[hr:IN_HYPEREDGE]->(h) SET hr.role='hub', hr.via=null
    WITH h, r
    UNWIND r.links AS l
    MATCH (m) WHERE id(m) = l.id
    MERGE (m)-[mr:IN_HYPEREDGE]->(h) SET mr.role='member', mr.via=l.via
    """, ns=NS, rows=payload)


def validate(nodes, rows):
    rel_of, repo_of = {}, {}
    for nid, (name, et, fp) in nodes.items():
        rest = (fp or "").replace("\\", "/")
        rest = rest[len(ROOT):] if rest.startswith(ROOT) else rest
        p = rest.split("/", 1)
        repo_of[nid], rel_of[nid] = (p[0], p[1]) if len(p) == 2 else ("?", rest)
    by_repo = collections.defaultdict(dict)
    for nid in nodes:
        by_repo[repo_of[nid]][rel_of[nid]] = nid
    lab = collections.Counter()
    base_pairs = base_pos = 0
    for rp in REPOS:
        out = subprocess.run(["git", "-C", ROOT + rp, "log", "--all",
                              "--pretty=format:\x01", "--name-only"],
                             capture_output=True, text=True, errors="replace").stdout
        for b in out.split("\x01")[1:]:
            ids = sorted({by_repo[rp][f] for f in
                          {l.strip() for l in b.splitlines() if l.strip()}
                          if f in by_repo[rp]})
            if 2 <= len(ids) <= 30:
                for a, c in itertools.combinations(ids, 2):
                    lab[(min(a, c), max(a, c))] += 1

    print("\nVALIDATION — does IDF weighting raise within-hyperedge precision?\n")
    print(f"{'metapath':<8}{'hubs':>6}{'pairs':>8}{'precision':>11}{'idf-weighted':>14}{'delta':>8}")
    for mp in ("P_R_P", "A_P_A", "A_P_R"):
        pw = collections.Counter()
        tw = collections.Counter()
        pu = tu = 0
        for r in rows:
            if r["metapath"] != mp:
                continue
            if mp == "A_P_R":
                acts = [m for m, v in r["via"].items() if v == "PERFORMS"]
                ress = [m for m, v in r["via"].items() if v != "PERFORMS"]
                pairs = [(a, c) for a in acts for c in ress]
            else:
                pairs = list(itertools.combinations(sorted(r["via"]), 2))
            for a, c in pairs:
                if repo_of[a] != repo_of[c]:
                    continue
                y = 1 if lab.get((min(a, c), max(a, c)), 0) else 0
                pu += y
                tu += 1
                pw[mp] += y * r["idf_weight"]
                tw[mp] += r["idf_weight"]
        if tu:
            up = pu / tu
            wp = pw[mp] / tw[mp] if tw[mp] > 0 else float("nan")
            nh = sum(1 for r in rows if r["metapath"] == mp)
            print(f"{mp:<8}{nh:>6}{tu:>8}{up:>11.3f}{wp:>14.3f}{wp-up:>+8.3f}")
    lows = sorted((r for r in rows if r["metapath"] == "P_R_P"),
                  key=lambda r: r["idf_weight"])[:3]
    print("\n  lowest-IDF hubs (the infrastructure the weighting demotes):")
    for r in lows:
        print(f"    idf {r['idf_weight']:<7} k={r['arity']-1:<4} {r['hub_name']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    drv = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with drv.session() as s:
            nodes, edges = load(s)
            rows = build(nodes, edges)
            cnt = collections.Counter(r["metapath"] for r in rows)
            print(f"built {len(rows)} hyperedges: " +
                  ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
            if not args.dry:
                emit(s, rows)
                # F93: verify by READ, never trust the write
                chk = list(s.run(
                    "MATCH (h:HyperedgeCandidate {namespace:$ns}) "
                    "RETURN h.metapath AS mp, count(*) AS c, "
                    "sum(CASE WHEN size([(x)-[:IN_HYPEREDGE]->(h) | x]) <> h.arity "
                    "THEN 1 ELSE 0 END) AS arity_mismatch", ns=NS))
                for r in chk:
                    print(f"  verified by read: {r['mp']} {r['c']} nodes, "
                          f"{r['arity_mismatch']} arity mismatches")
        if args.validate:
            validate(nodes, rows)
    finally:
        drv.close()


if __name__ == "__main__":
    main()
