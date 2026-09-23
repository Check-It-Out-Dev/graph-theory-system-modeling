"""The anchor bench: one HTML page the owner scores the calibration anchors on (published as a private artifact).

    PYTHONUTF8=1 python eval/put/judge/bench/build_bench.py <out.html>

Bundles, per anchor in judge/anchors.json: the task card, the diff, the test numbers, the applicable checks, the
work summary and both judge verdicts. The page stores the owner's scores (one document per anchor); they become
judge/owner-scores.json, which put_calibrate.py score reads.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PUT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PUT)
import put_paths    # noqa: E402
import put_contract  # noqa: E402
import put_judge    # noqa: E402

INSTANCE = "backend-conventions"
CRITERIA = ("correctness", "convention_fit", "design_fit", "test_quality", "graph_use")


def _j(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def bundle():
    tasks = {t["id"]: t for t in put_contract.tasks(INSTANCE)}
    anchors = _j(os.path.join(PUT, "judge", "anchors.json"))["anchors"]
    out = []
    for a in anchors:
        rd = os.path.join(put_paths.RUNS, *a["run"].replace("\\", "/").split("/"))
        t = tasks[a["task"]]
        v = (_j(os.path.join(rd, "verdict-r4.json")) or {}).get("verdict") or {}
        vr = (_j(os.path.join(rd, "verdict-r4-repeat.json")) or {}).get("verdict") or {}
        tests = _j(os.path.join(rd, "tests.json")) or {}
        checks = (_j(os.path.join(rd, "checks.json")) or {}).get("checks", {})
        with open(os.path.join(rd, "diff.patch"), encoding="utf-8") as f:
            diff = f.read()
        out.append({
            "id": a["id"], "run": a["run"].replace("\\", "/"), "task": a["task"], "rep": a["run"].rsplit(".r", 1)[-1],
            "title": t["title"], "text": t["text"], "interface": t.get("interface", []), "tags": t.get("tags", []),
            "diff": diff,
            "tests": {"build": tests.get("build_green"), "own": tests.get("own"), "own_classes": tests.get("own_classes"),
                      "hidden": tests.get("hidden"), "p2p": tests.get("pass_to_pass")},
            "checks": [{"rule": k, "passed": c["passed"], "value": c["value"], "seen": c["seen"]}
                       for k, c in checks.items() if c.get("applicable")],
            "work": put_judge.work_summary(_j(os.path.join(rd, "calls.json")) or []),
            "judge": v, "judge_repeat": {k: vr.get(k) for k in CRITERIA},
        })
    with open(os.path.join(PUT, "judge", "rubric-r4.md"), encoding="utf-8") as f:
        rubric = f.read()
    return {"anchors": out, "conventions": put_judge.canonical_rules(INSTANCE), "rubric": rubric}


def main(out_path):
    with open(os.path.join(HERE, "anchor-bench.template.html"), encoding="utf-8") as f:
        tpl = f.read()
    payload = json.dumps(bundle(), ensure_ascii=False).replace("</", "<\\/")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(tpl.replace("/*DATA*/null", payload))
    print(out_path)


if __name__ == "__main__":
    main(sys.argv[1])
