# MFQ cache invalidation for the curated re-clue batch (ErdosNavigator delta step 4).
# Reads eval/q/mfq_all.jsonl (NEVER modifies it) and the previous invalidation file, and emits
# eval/q/INVALIDATED_2026-09-02-curated.json.
#
# Two sets, per ledger L6 — caching against the primary set alone leaves stale answers live:
#   PRIMARY  = depends_on_subsystems intersects the curated changed set, OR the answer was
#              already invalidated by batch 2026-09-02 and has not been re-golded since.
#   ADVISORY = depends on a subsystem whose MEMBERSHIP held but whose CLUE BODY was corrected
#              in this batch (a cached answer quoting a corrected clause is stale prose even
#              though the graph under it did not move), plus the carried-forward advisory set.
#
# Usage: PYTHONUTF8=1 python mfq_invalidate_curated.py

import json, os

Q = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval", "q"))
BATCH = "2026-09-02-reclue"

# Curated changed set, as handed down with the task.
CHANGED = {4, 13, 17, 18, 2, 7, 16, 170, 171, 172, 173, 174, 175, 176, 177, 178}
# Bodies corrected this batch although membership held (see FIXES in nav_reclue_writer.py).
CORRECTED = {0, 1, 5, 6, 8, 9, 10}

# sub_id REMAP. mfq_all.jsonl stamps depends_on_subsystems against the 0-18 numbering, which
# predates the split and the merges; "depends on 17" no longer names anything navigable and
# "depends on 13" names a MERGED node. mfq_all.jsonl is NOT rewritten here (it is the frozen
# question bank); this table is the instruction for whoever re-golds.
REMAP = {
    17: [170, 171, 172, 173, 174, 175, 176, 177, 178],  # fan-out: needs re-execution
    13: [4],                                            # 1:1: mechanical rewrite
    18: [177],                                          # 1:1: mechanical rewrite
}

WHY = {
    4: "[4] absorbed ex-sub-13 (49 integration-test files): size 123->172, cohesion 0.362->0.478, "
       "cited entry PermissionUtils.java moved 73->68 external in-edges, clue body rewritten",
    13: "sub-13 was MERGED into [4] by curation batch 2026-09-02-curation; it holds no members, "
        "keeps role MERGED plus a [:SUPERSEDED_BY] edge, and its body is frozen pre-merge",
    17: "[17] became a GROUP of 9 children (170-178); its clue body was replaced by a routing "
        "body that deliberately carries no child detail - any answer that read [17] as a flat "
        "406-file subsystem must be re-derived from a child",
    18: "sub-18 was MERGED into child [177]; 147 of its 156 in-edges remain external, 9 became "
        "internal",
    2: "[2] retyped LAYER: the 'structurally a fragment, kept separate by clustering only' "
       "reading was REFUTED by measurement (86.2% fan-in, 8 consumers) and the merge candidacy "
       "is closed",
    7: "[7] retyped LAYER (91.1% fan-in across 15 consumers) without a membership change",
    16: "[16] retyped LAYER: the 'pure fragment held together by clustering' reading was REFUTED "
        "(93.7% fan-in, 12 consumers) and the merge candidacy is closed",
}
for c in range(170, 179):
    WHY[c] = f"child [{c}] of the frontend group is a new navigator with a first clue body; " \
             f"answers stamped against the flat [17] predate it"
FIX_WHY = {
    0: "[0] cited external_ratio corrected 0.939 -> 0.94",
    1: "[1] ungrounded 'feeds profile and company flows' replaced by the measured top seam to [4]",
    5: "[5] the '21 files' cascade-delete fan-out does not reproduce; corrected to a measured "
       "out-degree of 17",
    6: "[6] 'no actor roots' was false; InMemoryUserCache.java is one",
    8: "[8] the open MERGE CHECK caveat was resolved as KEEP by curation - a cached answer "
       "repeating it would queue closed work",
    9: "[9] 'third-largest subsystem' is now fourth after [4] grew to 172",
    10: "[10] the load-bearing seam list named 13->10, which no longer exists; re-measured as "
        "4->10 34/39",
}


def main():
    recs = [json.loads(l) for l in open(os.path.join(Q, "mfq_all.jsonl"), encoding="utf-8") if l.strip()]
    prev = json.load(open(os.path.join(Q, "INVALIDATED_2026-09-02.json"), encoding="utf-8"))
    prev_primary = {e["id"] for e in prev["invalidated"]}
    prev_advisory = {e["id"] for e in prev["advisory"]}

    primary, advisory = [], []
    for r in recs:
        dep = set(r.get("depends_on_subsystems") or [])
        hit = sorted(dep & CHANGED)
        reasons = [WHY[s] for s in hit if s in WHY]
        if r["id"] in prev_primary:
            reasons.append("carried forward: already invalidated by delta batch 2026-09-02 and "
                           "not re-golded since")
        if reasons:
            # sub_id is a foreign key (ledger L1) and this batch is the first time one MOVED,
            # so a fresh intersection is not enough for the re-golding step: it must know what
            # each stale stamp BECAME. 1:1 successors can be rewritten mechanically; a fan-out
            # successor cannot, because only re-executing the question reveals which children
            # actually hold its answer.
            succ, fanout = {}, False
            for s_ in sorted(dep):
                if s_ in REMAP:
                    succ[str(s_)] = REMAP[s_]
                    fanout = fanout or len(REMAP[s_]) > 1
            primary.append(dict(
                id=r["id"], q=r["q"], archetype=r.get("archetype"),
                stratum=r.get("stratum"), gold_status=r.get("gold_status"),
                depends_on_subsystems=sorted(dep), intersects_changed=hit,
                successor_subsystems=succ,
                remap_is_mechanical=bool(succ) and not fanout,
                requires_reexecution=fanout,
                gold_fingerprint_at_invalidation=r.get("gold_fingerprint"),
                reason=" | ".join(reasons)))
            continue
        fhit = sorted(dep & CORRECTED)
        areasons = [FIX_WHY[s] for s in fhit]
        if r["id"] in prev_advisory:
            areasons.append("carried forward from the 2026-09-02 advisory set (edge-touched "
                            "subsystem, structural/trophic archetype)")
        if areasons:
            advisory.append(dict(
                id=r["id"], q=r["q"], archetype=r.get("archetype"),
                depends_on_subsystems=sorted(dep), corrected_subsystems=fhit,
                reason=" | ".join(areasons)))

    out = dict(
        batch=BATCH, generated_by="ErdosNavigator/reclue", clue_version="curated-v2",
        source="mfq_all.jsonl", source_record_count=len(recs),
        changed_subsystems=sorted(CHANGED), corrected_subsystems=sorted(CORRECTED),
        sub_id_remap={str(k): v for k, v in REMAP.items()},
        remap_note="mfq_all.jsonl is deliberately NOT rewritten. 13->4 and 18->177 are 1:1 and "
                   "can be rewritten mechanically at re-gold time; 17->170-178 is a fan-out, so "
                   "those records carry requires_reexecution=true - only re-running the question "
                   "reveals which children hold its answer. Ledger L1 called sub_id a foreign "
                   "key; this batch is the first time one moved, so the key needs a documented "
                   "successor table rather than a fresh intersection.",
        policy="primary = depends_on_subsystems intersects the curated changed set {4, 13, 17, "
               "18, 2, 7, 16, 170-178}, OR the answer was already invalidated by batch "
               "2026-09-02; the runtime cache MUST drop these. advisory = depends on a "
               "subsystem whose membership held but whose clue body took a correction this "
               "batch, plus the carried-forward 2026-09-02 advisory set; recommended drop "
               "because the cached prose quotes a clause that is no longer in the clue. Ledger "
               "L6: caching against the primary set alone leaves stale answers live. Note that "
               "no mfq record can name 170-178 - those navigators did not exist when the bank "
               "was stamped, so every frontend answer reaches them through [17].",
        prev_primary_count=len(prev_primary), prev_advisory_count=len(prev_advisory),
        invalidated_count=len(primary), invalidated_ids=[e["id"] for e in primary],
        invalidated=primary,
        advisory_count=len(advisory), advisory_ids=[e["id"] for e in advisory],
        advisory=advisory)
    path = os.path.join(Q, "INVALIDATED_2026-09-02-curated.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    back = json.load(open(path, encoding="utf-8"))
    assert back["invalidated_count"] == len(back["invalidated"]) == len(primary)
    assert back["advisory_count"] == len(back["advisory"]) == len(advisory)
    assert not (set(back["invalidated_ids"]) & set(back["advisory_ids"])), "sets must be disjoint"
    src_mtime_ok = os.path.getsize(os.path.join(Q, "mfq_all.jsonl")) > 0
    print(f"VERIFIED BY READ: {path}")
    print(f"  {len(recs)} records in, {len(primary)} primary, {len(advisory)} advisory, "
          f"{len(recs) - len(primary) - len(advisory)} still valid; mfq_all.jsonl untouched={src_mtime_ok}")
    by_sub = {}
    for e in primary:
        for s in e["intersects_changed"]:
            by_sub[s] = by_sub.get(s, 0) + 1
    print("  primary hits by changed subsystem:", dict(sorted(by_sub.items())))


if __name__ == "__main__":
    main()
