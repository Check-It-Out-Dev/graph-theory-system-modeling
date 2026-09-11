# One-time FULL migration Neo4j(CheckItOutV3) -> Ladybug authoring store. After this
# runs green, nothing in codemap needs Neo4j; this script is the only file that may
# import the neo4j driver, kept for provenance and re-runs.
#
# Everything travels: entities WITH their three 4096-d embedding lenses, typed edges,
# ALGEBRA_VIOLATION, the curated navigation tree (Master -> Nav -> children/members),
# hyperedge cohorts, bi-temporal ClueSnapshots, CurationDecisions, DiagnosticState.
# nid = Neo4j internal id, preserved verbatim — every fingerprint and successor table
# in the repo keys on it (L11: vertex identity is the node id).
#
# Usage: PYTHONUTF8=1 python migrate_from_neo4j.py [--db <path>] [--wipe]

import argparse
import json
import os
import time

from neo4j import GraphDatabase

from ladybug_store import Store

BOLT, AUTH, NS = "bolt://127.0.0.1:7611", ("neo4j", "password"), "CheckItOutV3"
REL = ("IMPORTS|INJECTS|EXTENDS|CALLS|USES|PERFORMS|ACCESSES|IMPLEMENTS|MODIFIES|"
       "TRIGGERS|VALIDATES|AFFECTS|TESTED_BY|CONSTRAINS|APPLIES_IN|CONFIGURED_BY|INITIATES")
EMB = ("semantic_embedding", "behavioral_embedding", "structural_embedding")
ENT_COLS = ("name", "entity_type", "v4_subsystem", "file_path", "repo", "layer",
            "entry_point", "spine_membership", "local_height", "clue_version", "delta_batch")
NAV_COLS = ("name", "role", "routable", "parent", "size", "external_ratio", "ai_summary",
            "clue_version", "clue_body_status", "generated_by", "clue_delta_batch",
            "dossier_fingerprint", "spines", "entry_points", "contracts", "caveats",
            "responsibilities")


def jdump(x):
    return json.dumps(x, ensure_ascii=False, default=str)


def rest(props, taken):
    return jdump({k: v for k, v in props.items() if k not in taken})


_STORE = None


def create(conn_unused_store, table, data, pk):
    """Delegates to Store.create — the ONE armored writer (L11). Signature kept so the
    call sites read unchanged; first arg is the Store's connection owner."""
    _STORE.create(table, data, pk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--wipe", action="store_true", help="delete an existing store first")
    a = ap.parse_args()
    if a.wipe:
        from ladybug_store import DEFAULT_DB
        p = a.db or DEFAULT_DB
        for f in (p, p + ".wal"):
            if os.path.exists(f):
                os.remove(f)  # NOSONAR - operator's own path; see sonar-project.properties
    s = Store(a.db).init_schema()
    global _STORE
    _STORE = s
    drv = GraphDatabase.driver(BOLT, auth=AUTH)
    t0 = time.time()

    with drv.session() as n4:
        # ---- entities (with embeddings) --------------------------------------------
        nodes = n4.run(
            "MATCH (n:EntityDetail {namespace:$ns}) RETURN id(n) AS nid, "
            "properties(n) AS p", ns=NS).data()
        nid2sub = {r["nid"]: r["p"].get("v4_subsystem") for r in nodes}
        for r in nodes:
            p = r["p"]
            create(s.conn, "Entity", pk="nid", data=dict(
                nid=r["nid"], name=p.get("name"), entity_type=p.get("entity_type"),
                sub=p.get("v4_subsystem"), file_path=p.get("file_path"),
                repo=p.get("repo"), layer=p.get("layer"),
                # NOT `or ""` — that conflates False with absent (caught by the pack
                # rowset differential: 'False' cells became '')
                entry_point=("" if p.get("entry_point") is None
                             else str(p["entry_point"])),
                # Neo4j holds a LIST here; a list bound into a STRING column gets the
                # Kuzu-notation cast (['a'] -> '[a]', JSON destroyed) — same hazard
                # class as strings-that-look-like-literals, list-shaped. jdump it; the
                # [-prefix then routes it through the literal armor automatically.
                spine_membership=(jdump(p["spine_membership"])
                                  if isinstance(p.get("spine_membership"), list)
                                  else p.get("spine_membership")),
                local_height=(float(p["local_height"])
                              if p.get("local_height") is not None else None),
                clue_version=p.get("clue_version"), delta_batch=p.get("delta_batch"),
                sem_emb=p.get(EMB[0]), beh_emb=p.get(EMB[1]), str_emb=p.get(EMB[2]),
                props=rest(p, set(ENT_COLS) | set(EMB) | {"namespace"})))
        print(f"entities: {len(nodes)}  ({time.time()-t0:.0f}s)")

        # ---- typed dependency edges + violations -----------------------------------
        edges = n4.run(
            f"MATCH (x:EntityDetail {{namespace:$ns}})-[r:{REL}]->"
            "(y:EntityDetail {namespace:$ns}) RETURN id(x) AS aid, id(y) AS bid, "
            "type(r) AS t, r.source AS src", ns=NS).data()
        for e in edges:
            s.conn.execute(
                "MATCH (x:Entity {nid:$a}), (y:Entity {nid:$b}) "
                "CREATE (x)-[:Dep {rel:$t, source:$src}]->(y)",
                parameters=dict(a=e["aid"], b=e["bid"], t=e["t"], src=e["src"] or ""))
        viol = n4.run(
            "MATCH (x:EntityDetail {namespace:$ns})-[r:ALGEBRA_VIOLATION]->"
            "(y:EntityDetail {namespace:$ns}) RETURN id(x) AS aid, id(y) AS bid, "
            "r.source AS src", ns=NS).data()
        for e in viol:
            s.conn.execute(
                "MATCH (x:Entity {nid:$a}), (y:Entity {nid:$b}) "
                "CREATE (x)-[:Violation {source:$src}]->(y)",
                parameters=dict(a=e["aid"], b=e["bid"], src=e["src"] or ""))
        print(f"deps: {len(edges)}, violations: {len(viol)}  ({time.time()-t0:.0f}s)")

        # ---- navigation tree -------------------------------------------------------
        navs = n4.run(
            "MATCH (sn:SubsystemNavigator {namespace:$ns}) RETURN id(sn) AS xid, "
            "properties(sn) AS p", ns=NS).data()
        for r in navs:
            p = r["p"]
            create(s.conn, "Nav", pk="sub_id", data=dict(
                sub_id=int(p["sub_id"]), name=p.get("name"), role=p.get("role"),
                routable=bool(p.get("routable")),
                parent=(int(p["parent"]) if p.get("parent") is not None else None),
                size=(int(p["size"]) if p.get("size") is not None else None),
                external_ratio=(float(p["external_ratio"])
                                if p.get("external_ratio") is not None else None),
                ai_summary=p.get("ai_summary"), clue_version=p.get("clue_version"),
                clue_body_status=p.get("clue_body_status"),
                generated_by=p.get("generated_by"),
                clue_delta_batch=p.get("clue_delta_batch"),
                dossier_fingerprint=p.get("dossier_fingerprint"),
                spines=jdump(p.get("spines")), entry_points=jdump(p.get("entry_points")),
                contracts=jdump(p.get("contracts")), caveats=jdump(p.get("caveats")),
                responsibilities=jdump(p.get("responsibilities")),
                props=rest(p, set(NAV_COLS) | {"sub_id", "namespace"})))
        masters = n4.run(
            "MATCH (m:NavigationMaster {namespace:$ns}) RETURN properties(m) AS p",
            ns=NS).data()
        for i, r in enumerate(masters):
            p = r["p"]
            create(s.conn, "Master", pk="mid", data=dict(
                mid=i, ai_summary=p.get("ai_summary"),
                subsystem_index=p.get("subsystem_index"),
                global_caveats=jdump(p.get("global_caveats")),
                props=rest(p, {"ai_summary", "subsystem_index",
                               "global_caveats", "namespace"})))
        guides = n4.run(
            "MATCH (:NavigationMaster {namespace:$ns})-[:GUIDES]->"
            "(sn:SubsystemNavigator) RETURN sn.sub_id AS sid", ns=NS).data()
        for g in guides:
            s.conn.execute("MATCH (m:Master {mid:0}), (n:Nav {sub_id:$s}) "
                           "CREATE (m)-[:Guides]->(n)", parameters=dict(s=int(g["sid"])))
        child = n4.run(
            "MATCH (a:SubsystemNavigator {namespace:$ns})-[:GUIDES]->"
            "(b:SubsystemNavigator {namespace:$ns}) "
            "RETURN a.sub_id AS pa, b.sub_id AS ch", ns=NS).data()
        for g in child:
            s.conn.execute("MATCH (a:Nav {sub_id:$p}), (b:Nav {sub_id:$c}) "
                           "CREATE (a)-[:GuidesChild]->(b)",
                           parameters=dict(p=int(g["pa"]), c=int(g["ch"])))
        sup = n4.run(
            "MATCH (a:SubsystemNavigator {namespace:$ns})-[:SUPERSEDED_BY]->"
            "(b:SubsystemNavigator {namespace:$ns}) "
            "RETURN a.sub_id AS fa, b.sub_id AS fb", ns=NS).data()
        for g in sup:
            s.conn.execute("MATCH (a:Nav {sub_id:$a}), (b:Nav {sub_id:$b}) "
                           "CREATE (a)-[:SupersededBy]->(b)",
                           parameters=dict(a=int(g["fa"]), b=int(g["fb"])))
        members = n4.run(
            "MATCH (sn:SubsystemNavigator {namespace:$ns})-[:CONTAINS_MEMBER]->(e) "
            "RETURN sn.sub_id AS sid, id(e) AS nid", ns=NS).data()
        for m in members:
            s.conn.execute("MATCH (n:Nav {sub_id:$s}), (e:Entity {nid:$n2}) "
                           "CREATE (n)-[:Member]->(e)",
                           parameters=dict(s=int(m["sid"]), n2=m["nid"]))
        print(f"nav: {len(navs)} navs, {len(masters)} master, {len(guides)} guides, "
              f"{len(child)} child edges, {len(sup)} superseded, {len(members)} members "
              f"({time.time()-t0:.0f}s)")

        # ---- hyperedges ------------------------------------------------------------
        hyper = n4.run(
            "MATCH (h:HyperedgeCandidate {namespace:$ns}) RETURN id(h) AS xid, "
            "properties(h) AS p", ns=NS).data()
        for r in hyper:
            p = r["p"]
            create(s.conn, "Hyperedge", pk="key", data=dict(
                key=str(p.get("key") or r["xid"]), metapath=p.get("metapath"),
                hub_nid=p.get("hub_id"), hub_name=p.get("hub_name"),
                # hubsub was never a property of the hyperedge node — Neo4j readers
                # JOINed hub_id -> EntityDetail.v4_subsystem; the store materializes it
                hubsub=nid2sub.get(p.get("hub_id")),
                idf=(float(p["idf_weight"]) if p.get("idf_weight") is not None else None),
                arity=(int(p["arity"]) if p.get("arity") is not None else None),
                prior=(float(p["precision_prior"])
                       if p.get("precision_prior") is not None else None),
                members=jdump(p.get("member_names")), source=p.get("source") or "",
                props=rest(p, {"key", "metapath", "hub_id", "hub_name", "idf_weight",
                               "arity", "precision_prior", "member_names", "source",
                               "namespace"})))
        print(f"hyperedges: {len(hyper)}  ({time.time()-t0:.0f}s)")

        # ---- bi-temporal clue history + curation + diagnostics ---------------------
        snaps = n4.run("MATCH (c:ClueSnapshot {namespace:$ns}) RETURN id(c) AS xid, "
                       "properties(c) AS p", ns=NS).data()
        for r in snaps:
            p = r["p"]
            create(s.conn, "ClueSnap", pk="snap_id", data=dict(
                snap_id=str(r["xid"]),
                sub_id=(int(p["sub_id"]) if p.get("sub_id") is not None else None),
                taken_at=str(p.get("taken_at") or ""),
                t_created=str(p.get("t_created") or ""),
                t_expired=str(p.get("t_expired") or ""),
                superseded_by=str(p.get("superseded_by") or ""), body=jdump(p)))
        cur = n4.run("MATCH (c:CurationDecision {namespace:$ns}) RETURN id(c) AS xid, "
                     "properties(c) AS p", ns=NS).data()
        for r in cur:
            p = r["p"]
            create(s.conn, "CurationDecision", pk="did", data=dict(
                did=str(r["xid"]), batch=str(p.get("batch") or ""),
                decided_at=str(p.get("decided_at") or ""), body=jdump(p)))
        diag = n4.run("MATCH (d:DiagnosticState {namespace:$ns}) RETURN id(d) AS xid, "
                      "properties(d) AS p", ns=NS).data()
        for r in diag:
            p = r["p"]
            v = p.get("value")
            create(s.conn, "Diag", pk="oid", data=dict(
                oid=str(r["xid"]), functional=p.get("functional"), scope=p.get("scope"),
                value_num=(float(v) if isinstance(v, (int, float)) else None),
                value_json=(v if isinstance(v, str) else None),
                run_id=p.get("run_id"), computed_at=str(p.get("computed_at") or "")))
        print(f"snapshots: {len(snaps)}, curation: {len(cur)}, diag: {len(diag)}")

    drv.close()
    s.conn.execute("CREATE (:Meta {k:'migrated_at', v:$v})",
                   parameters=dict(v=time.strftime("%Y-%m-%dT%H:%M:%S")))
    s.conn.execute("CREATE (:Meta {k:'source', v:'neo4j:CheckItOutV3'})")
    s.conn.execute("CREATE (:Meta {k:'generation', v:'ladybug-v1'})")
    print("\nVERIFY BY READ:", json.dumps(s.counts(), indent=1))
    # JSON round-trip proof: the corruption class this migration was bitten by is
    # silent — a stored props/spines value that no longer json.loads is a FAIL.
    for row in s.q("MATCH (e:Entity) RETURN e.props AS p LIMIT 3"):
        json.loads(row["p"])
    for row in s.q("MATCH (n:Nav) RETURN n.spines AS sp, n.props AS p LIMIT 3"):
        json.loads(row["sp"])
        json.loads(row["p"])
    print("JSON round-trip: OK (Entity.props + Nav.spines/props parse back)")
    print(f"total {time.time()-t0:.0f}s -> {s.path}")


if __name__ == "__main__":
    main()
