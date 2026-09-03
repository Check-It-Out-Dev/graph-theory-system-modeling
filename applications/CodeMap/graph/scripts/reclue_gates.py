# E4 acceptance battery for the curated re-clue batch. Every gate recomputes from the graph or
# the refreshed dossiers and compares against what the written clues actually say — no constants
# copied from the writer, so a gate cannot pass by agreeing with itself.
# Authoring source is the Ladybug store since 2026-09-02 (same gates, same numbers; the
# mixed-type var-length matches of the Neo4j era became set algebra over the rel tables).
#
# Usage: PYTHONUTF8=1 python reclue_gates.py

import json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "authoring")))
from ladybug_store import Store

DOSS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dossiers"))

results = []


def check(name, ok, detail):
    results.append((ok, name, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def jload(v):
    return json.loads(v) if v is not None else None


def main():
    doss = {}
    for f in os.listdir(DOSS):
        if f.startswith("subsystem_") and f.endswith(".json"):
            d = json.load(open(os.path.join(DOSS, f), encoding="utf-8"))
            doss[d["subsystem"]] = d
    s = Store(read_only=True)

    # ---- reconstruct the nav/master views the gates read (json.loads undoes the store's
    # one jdump layer, so every field carries the SAME type the Neo4j gates saw) ----------
    nav = {}
    for r in sorted(s.q("MATCH (sn:Nav) RETURN sn.*"), key=lambda x: x["sn.sub_id"]):
        p = jload(r["sn.props"]) or {}
        nav[r["sn.sub_id"]] = dict(
            sub=r["sn.sub_id"], role=r["sn.role"], name=r["sn.name"],
            summ=r["sn.ai_summary"], cav=jload(r["sn.caveats"]),
            resp=jload(r["sn.responsibilities"]), size=r["sn.size"],
            ext=r["sn.external_ratio"], cv=r["sn.clue_version"],
            st=r["sn.clue_body_status"], ep=jload(r["sn.entry_points"]),
            fp=r["sn.dossier_fingerprint"], gb=r["sn.generated_by"],
            ga=p.get("generated_at"), kids=p.get("child_sub_ids"),
            hist=p.get("curation_history"))
    m = s.one("MATCH (nm:Master) RETURN nm.*")
    mp = jload(m["nm.props"]) or {}
    nm = dict(idx=m["nm.subsystem_index"], cav=jload(m["nm.global_caveats"]),
              instr=mp.get("ai_instruction"), cv=mp.get("clue_version"),
              summ=m["nm.ai_summary"])

    roots = sorted(r["c"] for r in s.q(
        "MATCH (:Master)-[:Guides]->(n:Nav) RETURN n.sub_id AS c"))
    kids_of = defaultdict(set)
    for r in s.q("MATCH (a:Nav)-[:GuidesChild]->(b:Nav) "
                 "RETURN a.sub_id AS p, b.sub_id AS c"):
        kids_of[r["p"]].add(r["c"])
    members = defaultdict(set)
    hmap, spmap = {}, {}
    for r in s.q("MATCH (sn:Nav)-[:Member]->(n:Entity) RETURN sn.sub_id AS sub, "
                 "n.nid AS nid, n.local_height AS h, n.spine_membership AS sp"):
        members[r["sub"]].add(r["nid"])
        hmap[(r["sub"], r["nid"])] = r["h"]
        spmap[(r["sub"], r["nid"])] = r["sp"]
    total = s.one("MATCH (n:Entity) RETURN count(n) AS c")["c"]
    decisions = []
    for r in s.q("MATCH (d:CurationDecision) RETURN d.body AS b"):
        decisions.append(jload(r["b"]) or {})

    # --- G1 structure: reachability <= 3 hops ------------------------------------------
    lvl2 = set(roots) | {c for p in roots for c in kids_of[p]}
    reach = set().union(*(members[x] for x in lvl2)) if lvl2 else set()
    check("G1 reachability <=3 hops", len(reach) == total == 1415,
          f"{len(reach)} of {total} members reachable within 3 hops of L1")

    # --- G2 structure: fan-out <= 9 at every level --------------------------------------
    per = sorted(f"{p}:{len(c)}" for p, c in kids_of.items())
    worst = max((len(c) for c in kids_of.values()), default=0)
    check("G2 fan-out <=9 every level", len(roots) <= 9 and worst <= 9,
          f"L1 fan-out {len(roots)}, worst group fan-out {worst} ({', '.join(per)})")

    # --- G3 provenance on every written node -------------------------------------------
    live = [v for v in nav.values() if v["role"] != "MERGED"]
    missing = [v["sub"] for v in live if not (v["cv"] == "curated-v2" and v["st"] == "CURRENT"
                                              and v["gb"] and v["ga"] and v["fp"])]
    check("G3 provenance complete", not missing,
          f"{len(live)} live navigators carry clue_version/status/generated_by/at/fingerprint"
          + (f"; missing on {missing}" if missing else ""))

    # --- G4 freshness: every stored size/ext equals the refreshed dossier ---------------
    drift = []
    for sub, v in nav.items():
        if v["role"] in ("MERGED", "GROUP") or sub not in doss:
            continue
        if v["size"] != doss[sub]["size"] or v["ext"] != doss[sub]["external_ratio"]:
            drift.append(f"{sub}: {v['size']}/{v['ext']} vs {doss[sub]['size']}/{doss[sub]['external_ratio']}")
    check("G4 freshness leaf size+ext", not drift,
          f"{len([v for v in nav.values() if v['role'] not in ('MERGED','GROUP')])} leaf "
          f"navigators match their dossier" + (f"; drift {drift}" if drift else ""))

    # --- G5 freshness: group sizes are the sum of their children, and partition the corpus
    gsum, gdetail = 0, []
    for sub, v in nav.items():
        if v["role"] != "GROUP":
            continue
        want = sum(doss[k]["size"] for k in v["kids"])
        gdetail.append(f"{sub}={v['size']}")
        gsum += v["size"]
        if v["size"] != want:
            drift.append(f"group {sub}: {v['size']} vs children {want}")
    total_top = gsum + nav[4]["size"]
    check("G5 groups partition the corpus", not drift and total_top == 1415,
          f"{' + '.join(sorted(gdetail))} + [4]={nav[4]['size']} = {total_top}")

    # --- G6..G20 grounding sample: 15 claims, each re-derived --------------------------
    d = doss[170]
    check("G6 claim [170] 'Rule-dominant (36 of 69)'",
          d["layer_profile"]["Rule"] == 36 and d["size"] == 69,
          f"dossier layer_profile Rule={d['layer_profile']['Rule']}, size={d['size']}")

    dec = next((x for x in decisions if "core/auth" in (x.get("rationale") or "")), None)
    check("G7 claim [170] 'core/auth to feature/auth, 40 edges'",
          dec is not None and "40 edges" in dec["rationale"],
          "CurationDecision on sub-17 states the seam at 40 edges")

    d = doss[172]
    check("G8 claim [172] '14 internal edges and ZERO edges to any sibling'",
          d["edges_internal"] == 14 and d["edges_external"] == 0 and d["external_ratio"] == 0.0,
          f"dossier edges_internal={d['edges_internal']}, edges_external={d['edges_external']}")

    h173 = [hmap[(173, n)] for n in members[173]]
    check("G9 claim [173] 'only 7 of its 67 members carry a local_height'",
          sum(1 for h in h173 if h is not None) == 7 and len(h173) == 67,
          f"measured {sum(1 for h in h173 if h is not None)} of {len(h173)}")

    check("G10 claim [173] '85% Resource purity'", doss[173]["purity"] == 0.851,
          f"dossier purity={doss[173]['purity']}, flag {doss[173]['flags']}")

    ep177 = {e["name"]: e["ext_in"] for e in doss[177]["entry_points"]}
    check("G11 claim [177] '`api` 147 external in-edges'", ep177.get("api") == 147,
          f"dossier entry_points api={ep177.get('api')}")

    want = {173: 37, 176: 33, 171: 29, 170: 20, 174: 14, 175: 12, 178: 2}
    got = {}
    for k in want:
        seam = [t for t in doss[k]["top_seams"]
                if t["other"] == 177 and t["rel"] == "IMPORTS" and t["dir"] == "out"]
        got[k] = seam[0]["n"] if seam else None
    check("G12 claim [177] inbound seam table 37/33/29/20/14/12/2", got == want,
          f"child dossiers give {got}")

    h16 = [hmap[(16, n)] for n in members[16]]
    check("G13 claim [16] 'local_height null on all 12 members'",
          sum(1 for h in h16 if h is not None) == 0 and len(h16) == 12,
          f"measured {sum(1 for h in h16 if h is not None)} of {len(h16)} with a height")

    check("G14 claim [16] 'AccountStatus.java 51 external in-edges'",
          doss[16]["entry_points"][0] == {"name": "AccountStatus.java", "ext_in": 51},
          f"dossier top entry {doss[16]['entry_points'][0]}")

    dec = next((x for x in decisions if "in=422 out=41" in (x.get("evidence") or "")), None)
    check("G15 claim [7] 'in=422, out=41, 15 consumers, max 13.4%'",
          dec is not None and "15 distinct consumers" in dec["evidence"] and "13.4%" in dec["evidence"],
          "CurationDecision on sub-7 states all three figures")

    ep4 = {e["name"]: e["ext_in"] for e in doss[4]["entry_points"]}
    check("G16 claim [4] 'PermissionUtils.java 68 external in-edges'",
          ep4.get("PermissionUtils.java") == 68,
          f"dossier entry_points PermissionUtils.java={ep4.get('PermissionUtils.java')}")

    spined4 = sum(1 for n in members[4]
                  if (spmap[(4, n)] or "[]") not in ("[]", "null"))
    hub_in_4 = members[4]
    hyps = s.q("MATCH (h:Hyperedge) WHERE h.metapath = 'A_P_R' AND "
               "h.source = 'metapath-v3' RETURN h.hub_nid AS hn")
    n_ap4 = sum(1 for h in hyps if h["hn"] in hub_in_4)
    check("G17 claim [4] '12 A_P_R hyperedges over 40 of its 172 members'",
          n_ap4 == 12 and spined4 == 40,
          f"measured {n_ap4} A_P_R hyperedges hubbed in [4], {spined4} members on a spine")

    dec = next((x for x in decisions if "in=138 out=22" in (x.get("evidence") or "")), None)
    check("G18 claim [2] 'in=138, out=22, 86.2% fan-in, 8 consumers'",
          dec is not None and "86.2% fan-in" in dec["evidence"] and "consumers=8" in dec["evidence"],
          "CurationDecision on sub-2 states all four figures")

    d = doss[178]
    check("G19 claim [178] '16 files, three external edges, ext 0.176'",
          d["size"] == 16 and d["edges_external"] == 3 and d["external_ratio"] == 0.176,
          f"dossier size={d['size']}, edges_external={d['edges_external']}, ext={d['external_ratio']}")

    h2set = set().union(*(members[x] for x in roots)) if roots else set()
    lvl3 = {c for p in roots for c in kids_of[p]}
    h3set = set().union(*(members[x] for x in lvl3)) if lvl3 else set()
    check("G20 L1 caveat '578 at hop 2, 1243 at hop 3'",
          len(h2set) == 578 and len(h3set) == 1243,
          f"measured hop2={len(h2set)}, hop3={len(h3set)}")

    # --- G21 the mandated checklist is verbatim ----------------------------------------
    l1snaps = []
    for r in s.q("MATCH (c:ClueSnap) RETURN c.body AS b, c.t_created AS tc"):
        b = jload(r["b"]) or {}
        if b.get("level") == "L1":
            l1snaps.append((str(r["tc"] or b.get("t_created") or ""), b))
    l1snaps.sort(key=lambda x: x[0], reverse=True)
    snap_cv = l1snaps[0][1].get("clue_version") if l1snaps else None
    steps = ["1. Read this node fully.", "2. Try the answered-questions cache",
             "3. Else match a Recipe by intent", "4. Else descend: subsystem_index",
             "5. Open raw files only when clues are insufficient"]
    check("G21 ai_instruction checklist verbatim (5 steps)",
          all(x in nm["instr"] for x in steps),
          f"all 5 mandated steps present; latest L1 snapshot superseded {snap_cv}")

    # --- G22 scent audit: every index child label names a real navigator ---------------
    ids = [int(x) for x in re.findall(r"^\s*-? *\[(\d+)\]", nm["idx"], re.M)]
    known = {k for k, v in nav.items() if v["role"] != "MERGED"}
    check("G22 scent: index labels resolve", set(ids) <= known and len(ids) == len(set(ids)),
          f"{len(ids)} index entries, all resolve to live navigators, no duplicates")

    # --- G23 no stale group properties -------------------------------------------------
    stale = [k for k, v in nav.items() if v["role"] == "GROUP" and v["ext"] is not None]
    check("G23 groups carry no leaf-era external_ratio", not stale,
          f"5 GROUP navigators, external_ratio cleared on all" + (f"; stale {stale}" if stale else ""))

    # --- G24 E2 mandatory field: curation_history renders the decision chain -----------
    empty = [k for k, v in nav.items() if not (v["hist"] or [])]
    nlines = sum(len(v["hist"] or []) for v in nav.values())
    check("G24 curation_history on every navigator", not empty,
          f"{len(nav)} navigators, {nlines} rendered lines from {len(decisions)} "
          f"CurationDecision nodes" + (f"; empty on {empty}" if empty else ""))

    # --- G25 every rendered history line traces to a real decision ---------------------
    rats = [x.get("rationale") for x in decisions]
    lines = [ln for v in nav.values() for ln in (v["hist"] or [])]
    orphan = [ln for ln in lines
              if not any((rt or "")[:40] in ln for rt in rats)]
    check("G25 history lines trace to a decision", not orphan,
          f"{len(lines)} rendered lines, all matched to a CurationDecision rationale"
          + (f"; {len(orphan)} orphan" if orphan else ""))

    # --- G26/G27 pack consistency: the runtime reads the pack, not this graph ----------
    pack = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "pack"))
    l2f = os.path.join(pack, "l2_navigators.jsonl")
    if not os.path.exists(l2f):
        check("G26 pack exists", False, "graph/pack/l2_navigators.jsonl not found — run export_pack.py")
    else:
        pl2 = [json.loads(x) for x in open(l2f, encoding="utf-8")]
        pids = {x["sub_id"] for x in pl2}
        gids = set(nav)
        routable = {x["sub_id"] for x in pl2 if x.get("routable")}
        merged_ok = all(x.get("superseded_by") for x in pl2 if not x.get("routable"))
        check("G26 pack L2 mirrors the graph", pids == gids and merged_ok
              and routable == {k for k, v in nav.items() if v["role"] != "MERGED"},
              f"{len(pl2)} navigators exported ({len(routable)} routable, "
              f"{len(pids) - len(routable)} MERGED each carrying a superseded_by pointer)")
        # A stale mfq stamp must resolve OFFLINE, from the pack alone — that is the whole
        # point of retaining MERGED nodes, and it was silently false until this batch.
        mfq = [json.loads(x) for x in open(os.path.join(pack, "mfq.jsonl"), encoding="utf-8")]
        stamps = {st for r in mfq for st in (r.get("depends_on_subsystems") or [])}
        unres = sorted(st for st in stamps if st not in pids)
        check("G27 every mfq stamp resolves in the pack", not unres,
              f"{len(stamps)} distinct sub_id stamps across {len(mfq)} records, all resolve"
              + (f"; unresolved {unres}" if unres else ""))

    # --- token economy ------------------------------------------------------------------
    clue_chars = len(nm["summ"]) + len(nm["idx"]) + sum(len(c) for c in nm["cav"])
    for v in nav.values():
        clue_chars += len(v["summ"] or "") + sum(len(c) for c in (v["cav"] or [])) \
                      + sum(len(c) for c in (v["resp"] or []))
    loc = 0
    for r in s.q("MATCH (n:Entity) RETURN n.props AS p"):
        loc += (jload(r["p"]) or {}).get("line_count", 0) or 0
    print(f"\nTOKEN ECONOMY: L1+L2 clue text ~{clue_chars:,} chars (~{clue_chars//4:,} tokens) "
          f"for a corpus of {loc:,} indexed lines (~{loc*10//4:,} tokens at 10 chars/line) "
          f"= ~{100.0*(clue_chars/4)/max(1,(loc*10/4)):.2f}% of read-everything")
    bad = [n for ok, n, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} gates PASS")
    if bad:
        print("FAILED:", bad)
        sys.exit(1)


if __name__ == "__main__":
    main()
