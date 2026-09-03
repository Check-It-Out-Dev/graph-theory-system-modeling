# Owner decision Q1=B (2026-09-02): promote the index groups to real GROUP navigators.
# Elegance note: sub-17 (FE) and sub-4 (opportunities) already ARE their groups, so only
# 4 new GROUP nodes are created; Master fan-out becomes exactly 6 and every member stays
# within 3 hops (master -> group -> subsystem -> member). Also records owner decisions
# Q1/Q2 as CurationDecision rows. Verifies all writes by read.
#
# STORE PORT (2026-09-02): Neo4j -> Ladybug Store. GUIDES(master->nav) is the Guides rel
# table, GUIDES(nav->nav) is GuidesChild; the (V3Master)-[:HAS_DECISION]-> edge is DROPPED
# (decisions are standalone rows; `batch`/`did` carry the linkage). --dry added.
#
# Usage: PYTHONUTF8=1 python c2b_groups.py [--dry]

import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "authoring")))
from ladybug_store import Store

GROUPS = [  # (sub_id, name, member top-level sub_ids)
    (201, "GROUP Billing, identity & notifications", [11, 10, 12]),
    (202, "GROUP Platform runtime & shared layers", [3, 7, 0, 16, 2]),
    (203, "GROUP Authentication & validation", [9, 6, 8]),
    (204, "GROUP Data, deletion & support", [1, 5, 14, 15]),
]
DIRECT = [17, 4]  # already group-shaped; stay directly under the master
BATCH = "2026-09-02-curation"

DECIDED = [
    ("Q1 master fan-out", "B: 6-way top level (4 new GROUPs; 17 and 4 already group-shaped)",
     "uniform fan-out <=9 at every level; members stay <=3 hops; C4 discipline applied to ourselves"),
    ("Q2 real-login.ts placement", "A: stays in child 173 (e2e harness)",
     "tier coherence beats a degree-0 affinity; dissent already recorded on the node"),
]


def main():
    dry = "--dry" in sys.argv
    s = Store(read_only=dry)
    now = _dt.datetime.now().isoformat()
    if dry:
        have = {r["sid"] for r in s.q("MATCH (n:Nav) RETURN n.sub_id AS sid")}
        for gid, name, members in GROUPS:
            print(f"DRY group {gid} '{name}': "
                  f"{'exists' if gid in have else 'CREATE'}, adopt {sorted(members)}, "
                  f"drop master->child Guides for those")
        print(f"DRY: record {len(DECIDED)} owner decisions as CurationDecision rows")
        return

    mid = s.one("MATCH (nm:Master) RETURN nm.mid AS mid")["mid"]
    have = {r["sid"] for r in s.q("MATCH (n:Nav) RETURN n.sub_id AS sid")}
    for gid, name, members in GROUPS:
        if gid not in have:
            s.create("Nav", dict(sub_id=gid, name=name, role="GROUP", routable=True,
                                 clue_version="curated-v1"), "sub_id")
            s.merge_props("Nav", "sub_id", gid, dict(
                clue_body_status="MISSING", created_by="c2b_groups/Q1-B", created_at=now))
        s.conn.execute(
            "MATCH (nm:Master), (g:Nav) WHERE nm.mid = $mid AND g.sub_id = $gid "
            "MERGE (nm)-[:Guides]->(g)", parameters=dict(mid=mid, gid=gid))
        for t in sorted(members):
            s.conn.execute(
                "MATCH (g:Nav), (t:Nav) WHERE g.sub_id = $gid AND t.sub_id = $t "
                "MERGE (g)-[:GuidesChild]->(t)", parameters=dict(gid=gid, t=t))
            s.conn.execute(
                "MATCH (nm:Master)-[r:Guides]->(t:Nav) WHERE t.sub_id = $t DELETE r",
                parameters=dict(t=t))
    for i, (q, choice, why) in enumerate(DECIDED):
        s.create("CurationDecision", dict(
            did=f"c2b-{BATCH}-q{i + 1}", batch=BATCH, decided_at=now,
            body=json.dumps(dict(
                subsystem=q, action="OWNER_DECIDED", target=choice, rationale=why,
                evidence="curator report Q1/Q2 numbers", decided_by="owner:Norbert"),
                ensure_ascii=False)), "did")

    # verify by read: top fan-out + 3-hop reach (master -> top -> child -> member)
    top = sorted(r["sid"] for r in s.q(
        "MATCH (:Master)-[:Guides]->(t:Nav) RETURN t.sub_id AS sid"))
    kids = {}
    for r in s.q("MATCH (a:Nav)-[:GuidesChild]->(b:Nav) "
                 "RETURN a.sub_id AS a, b.sub_id AS b"):
        kids.setdefault(r["a"], set()).add(r["b"])
    role = {r["sid"]: r["role"] for r in s.q(
        "MATCH (n:Nav) RETURN n.sub_id AS sid, n.role AS role")}
    members = {}
    for r in s.q("MATCH (x:Nav)-[:Member]->(n:Entity) "
                 "RETURN x.sub_id AS sid, n.nid AS nid"):
        members.setdefault(r["sid"], set()).add(r["nid"])
    frontier, seen = set(top), set(top)
    for _ in range(2):
        frontier = {k for f in frontier for k in kids.get(f, ())} - seen
        seen |= frontier
    reach = set()
    for sid in seen:
        if role.get(sid) not in ("MERGED", "GROUP"):
            reach |= members.get(sid, set())
    print(f"master children: {top}")
    print(f"members reachable <=3 hops from master: {len(reach)}")
    assert top == sorted([g[0] for g in GROUPS] + DIRECT)
    assert len(reach) == 1415
    dcount = sum(1 for r in s.q("MATCH (d:CurationDecision) RETURN d.body AS b")
                 if json.loads(r["b"] or "{}").get("action") == "OWNER_DECIDED")
    print(f"owner decisions recorded: {dcount}")
    print("Q1-B structure VERIFIED")


if __name__ == "__main__":
    main()
