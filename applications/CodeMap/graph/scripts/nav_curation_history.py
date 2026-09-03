# Completes the E2 contract for re-clue batch 2026-09-02-reclue: the mandatory
# `curation_history` field ("the CurationDecision chain, one line each").
#
# This is a MECHANICAL RENDER, not authored prose — every line is assembled from
# CurationDecision fields, so it is grounded by construction and cannot drift from the
# decision record. In the store a decision is one row whose `body` JSON carries
#   subsystem (string key) · action · target · rationale · evidence · decided_by
# There is no `kind` field; `action` is the decision type.
#
# Same batch as nav_reclue_writer.py and the same clue_version, so this is a completion of
# that write rather than a new version of it: NO new ClueSnap is created. Snapshotting a
# field addition inside one batch would inflate the bi-temporal record with versions that
# were never published.
#
# STORE PORT (2026-09-02): Neo4j -> Ladybug Store. curation_history lands in Nav props
# (it is a long-tail annotation, not a typed column); rows are pulled sorted by did so the
# rendered line order is deterministic.
#
# Usage: PYTHONUTF8=1 python nav_curation_history.py [--dry]

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "authoring")))
from ladybug_store import Store

BATCH = "2026-09-02-reclue"

# Decisions that apply to a navigator beyond the one keyed by its own sub_id.
# The nine frontend children exist BECAUSE of the sub-17 split, so they all inherit it.
EXTRA = {c: ["17"] for c in range(170, 179)}
EXTRA[4] = ["13"]            # absorbed sub-13
EXTRA[177] = ["17", "18"]    # created by the split, then absorbed sub-18
EXTRA[173] = ["17", "Q2 real-login.ts placement"]
for g in (201, 202, 203, 204):
    EXTRA[g] = ["Q1 master fan-out"]
EXTRA[17] = ["Q1 master fan-out"]
# The ASSIGNMENT-REVIEW decision (subsystem '-1') confirmed all 14 flagged nodes; attach it to
# the navigators its evidence actually names, so the line is never a generic footer.
for sub in (3, 8, 15, 173, 174, 176, 177):
    EXTRA.setdefault(sub, []).append("-1")


def one_line(d, limit=260):
    """action + target + the load-bearing opening of the rationale, trimmed at a sentence."""
    txt = (d.get("rationale") or "").strip()
    if len(txt) > limit:
        cut = txt.rfind(". ", 0, limit)
        txt = txt[:cut + 1] if cut > 60 else txt[:limit].rstrip() + "..."
    head = f"[{d.get('action')}]"
    tgt = (d.get("target") or "").strip()
    if tgt and tgt not in txt:
        head += f" {tgt} —"
    who = d.get("decided_by") or "GrothendieckV5"
    return f"{head} {txt} (decided_by {who})"


def main():
    dry = "--dry" in sys.argv
    s = Store(read_only=dry)
    rows = s.q("MATCH (d:CurationDecision) RETURN d.did AS did, d.body AS body")
    rows.sort(key=lambda r: str(r["did"]))        # determinism at the boundary
    decs = {}
    for r in rows:
        d = json.loads(r["body"] or "{}")
        decs.setdefault(str(d.get("subsystem")), []).append(d)
    print(f"CurationDecision rows: {sum(len(v) for v in decs.values())} across "
          f"{len(decs)} keys")

    navs = s.q("MATCH (sn:Nav) RETURN sn.sub_id AS sub, sn.role AS role")
    navs.sort(key=lambda n: n["sub"])
    written, empty = 0, []
    for n in navs:
        sub = n["sub"]
        keys = [str(sub)] + EXTRA.get(sub, [])
        lines, seen = [], set()
        for k in keys:
            for d in decs.get(k, []):
                ln = one_line(d)
                if ln not in seen:
                    seen.add(ln)
                    lines.append(ln)
        if not lines:
            empty.append(sub)
            continue
        if dry:
            print(f"DRY sub-{sub}: {len(lines)} decision line(s)")
            continue
        s.merge_props("Nav", "sub_id", sub, dict(
            curation_history=lines, curation_history_batch=BATCH))
        written += 1
    if dry:
        print(f"DRY: would write {len(navs) - len(empty)} nodes; no decision for {empty}")
        return
    chk = s.q("MATCH (sn:Nav) RETURN sn.sub_id AS sub, sn.role AS role, sn.props AS props")
    chk.sort(key=lambda r: r["sub"])
    print("VERIFIED BY READ:")
    missing = []
    for r in chk:
        nlines = len(json.loads(r["props"] or "{}").get("curation_history") or [])
        print(f"  sub-{r['sub']:<3} {str(r['role']):<7} {nlines} curation_history line(s)")
        if nlines == 0:
            missing.append(r["sub"])
    assert not missing, f"navigators without curation_history: {missing}"
    print(f"OK: {written} navigators carry a rendered CurationDecision chain")


if __name__ == "__main__":
    main()
