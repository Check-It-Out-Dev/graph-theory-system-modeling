"""Version drift: what the graph answers differently after a pack change, measured without a model.

    python graph/delta/drift.py --old graph/pack --new delta/pack.next [--out drift.json] [--invalidated pack/INVALIDATED_delta.json]

Every bank question whose archetype maps onto an engine verb (locate → find, flow, boundary → seam,
onboarding → enter, trophic_inversion → spine, overview → map, health → coupling + hubs) is executed
by the engine on both packs; the canonical result (affordances and the echoed DSL stripped) is hashed.
`drift_rate` = differing rows / compared rows, excluding rows the delta invalidated on purpose.
Deterministic, seconds, no secret — so it runs in the decision job before the Release is published
and its artifact rides the ledger (`graph/ledger/<version>.drift.json`); the nightly quality run
publishes the latest rate as `codemap_quality_version_drift_rate`.
"""

import argparse
import hashlib
import json
import os
import sys

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(R, "app"))

VERB_BY_ARCHETYPE = {
    "locate": lambda p: [("find", [p.get("rx", "")])] if p and p.get("rx") else [],
    "flow": lambda p: [("flow", [p.get("rx", ""), str(p.get("hops", 1))])] if p and p.get("rx") else [],
    "boundary": lambda p: [("seam", [str(p.get("a")), str(p.get("b"))])] if p and p.get("a") is not None else [],
    "onboarding": lambda p: [("enter", [str(p.get("sub"))])] if p and p.get("sub") is not None else [],
    "trophic_inversion": lambda p: [("spine", [str(p.get("sub"))])] if p and p.get("sub") is not None else [],
    "overview": lambda p: [("map", [])],
    "health": lambda p: [("health", ["coupling"]), ("health", ["hubs"])],
}


def canonical(result):
    if isinstance(result, dict):
        return {k: canonical(v) for k, v in sorted(result.items()) if k not in ("affordances", "dsl", "done")}
    if isinstance(result, list):
        # order is not meaning: a rebuilt LadybugDB may return the same hits in another order
        return sorted((canonical(x) for x in result), key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
    return result


def digest(result):
    return hashlib.sha256(json.dumps(canonical(result), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def execute(engine, verb, args):
    try:
        return digest(getattr(engine, verb)(*args)), None
    except Exception as ex:  # a DslError on one pack is itself a difference worth recording
        return "error", f"{type(ex).__name__}: {str(ex)[:120]}"


def compare(old_pack, new_pack, invalidated=(), use_ladybug=True):
    from engine import Engine
    old, new = Engine(use_ladybug=use_ladybug, pack_dir=old_pack), Engine(use_ladybug=use_ladybug, pack_dir=new_pack)
    bank = [json.loads(l) for l in open(os.path.join(new_pack, "mfq.jsonl"), encoding="utf-8") if l.strip()]
    skip = set(invalidated) | set(old.invalidated) | set(new.invalidated)
    rows, compared, drifted = [], 0, 0
    for q in bank:
        plan = VERB_BY_ARCHETYPE.get(q.get("archetype"), lambda p: [])(q.get("recipe_params") or {})
        if not plan:
            continue
        if q["id"] in skip:
            rows.append({"id": q["id"], "archetype": q["archetype"], "status": "invalidated"})
            continue
        diffs, executable = [], False
        for verb, args in plan:
            a, ea = execute(old, verb, args)
            b, eb = execute(new, verb, args)
            executable = executable or a != "error" or b != "error"
            if a != b:
                diffs.append({"dsl": f"{verb}({', '.join(args)})", "old": a, "new": b, "old_error": ea, "new_error": eb})
        if not executable:  # the recipe's regex is not an entity the verb accepts on either pack
            rows.append({"id": q["id"], "archetype": q["archetype"], "status": "not_executable"})
            continue
        compared += 1
        drifted += 1 if diffs else 0
        rows.append({"id": q["id"], "archetype": q["archetype"], "status": "drifted" if diffs else "stable", "diffs": diffs})
    return {"schema": 1, "old_pack": os.path.abspath(old_pack), "new_pack": os.path.abspath(new_pack),
            "old_version": _version(old_pack), "new_version": _version(new_pack),
            "compared": compared, "drifted": drifted, "invalidated": sum(1 for r in rows if r["status"] == "invalidated"),
            "not_executable": sum(1 for r in rows if r["status"] == "not_executable"),
            "drift_rate": round(drifted / compared, 4) if compared else 0.0,
            "drifted_ids": [r["id"] for r in rows if r["status"] == "drifted"], "rows": rows}


def _version(pack):
    p = os.path.join(pack, "manifest.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8")).get("pack_version")
        except ValueError:
            return None
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--invalidated", default=None, help="a JSON file with {invalidated: [ids]} to exclude on top of the packs' own")
    ap.add_argument("--no-ladybug", action="store_true")
    a = ap.parse_args(argv)
    inv = []
    if a.invalidated and os.path.exists(a.invalidated):
        inv = json.load(open(a.invalidated, encoding="utf-8")).get("invalidated", [])
    rep = compare(a.old, a.new, inv, use_ladybug=not a.no_ladybug)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(rep, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print(f"drift {rep['old_version']} → {rep['new_version']}: {rep['drifted']}/{rep['compared']} rows differ "
          f"(rate {rep['drift_rate']:.3f}), {rep['invalidated']} invalidated on purpose"
          + (": " + ", ".join(rep["drifted_ids"][:12]) if rep["drifted_ids"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
