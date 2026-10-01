"""Check the answer keys in `eval/erdos/gold/<problem>.json` before a judge reads them.

    PYTHONUTF8=1 python eval/erdos/take_gold.py [--workspace C:/Users/Norbert/erdos-ws]

Each key was built by a read-only agent from the source files only (no graph tool), so it favours
neither arm. A key must carry every field the judge reads, and every `must_find` path must exist in the
workspace checkouts the arms work in. Prints one line per key; exits 1 when a key is incomplete, a path
is absent, or a problem has no key.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold")
FIELDS = ("id", "valid", "validity_note", "must_find", "key_facts", "gaps", "invariants", "good_designs", "red_flags", "architecture")
ARCH_FIELDS = ("pattern", "where", "today", "fit", "breaks")


def load(problem_id):
    path = os.path.join(GOLD, f"{problem_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check(key, workspace):
    """-> list of problems with the key (empty = usable)."""
    problems = [f"missing field {f}" for f in FIELDS if f not in key]
    for m in key.get("must_find") or []:
        if not os.path.exists(os.path.join(workspace, m["path"])):
            problems.append(f"absent path {m['path']}")
    for entry in key.get("architecture") or []:
        problems += [f"architecture entry without {f}" for f in ARCH_FIELDS if not entry.get(f)]
        for path in entry.get("where") or []:
            if not os.path.exists(os.path.join(workspace, path)):
                problems.append(f"absent architecture path {path}")
    return problems


def main(argv=None):
    sys.path.insert(0, HERE)
    import run_pairs
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="C:/Users/Norbert/erdos-ws")
    a = ap.parse_args(argv)
    bad = 0
    for p in run_pairs.load_problems():
        key = load(p["id"])
        issues = ["no key"] if key is None else check(key, a.workspace)
        bad += bool(issues)
        counts = "" if key is None else (f"valid={key.get('valid')} must_find={len(key.get('must_find') or [])} "
                                         f"key_facts={len(key.get('key_facts') or [])} gaps={len(key.get('gaps') or [])} "
                                         f"architecture={len(key.get('architecture') or [])}")
        print(f"{p['id']}: {counts} {'OK' if not issues else issues}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
