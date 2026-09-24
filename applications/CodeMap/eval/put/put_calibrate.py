"""The judge against a human: anchors chosen from real runs, scored by the owner, compared criterion by criterion.

    PYTHONUTF8=1 python eval/put/put_calibrate.py select --label <baseline label> [--n 8]
    PYTHONUTF8=1 python eval/put/put_calibrate.py sheet                     # what the owner reads before scoring
    PYTHONUTF8=1 python eval/put/put_calibrate.py score                     # the judge vs the owner -> judge/calibration.json

Anchors are baseline runs spread over the judge's range (sorted by the judge's mean score, taken at even steps), so
the comparison covers good and poor changes alike. The owner scores the criteria a machine cannot settle (convention
fit, design fit, test quality) on the rubric's own 1-5 scale; correctness and graph use have deterministic signals
beside them (the hidden tests, the graph_first check) and are compared with those. Per criterion, binarised at >= 4:
agreement, prevalence, Cohen's kappa, Gwet's AC1; Spearman's rho on the raw scores (METRICS.md section 7).
"""

import argparse
import json
import os
import shutil
import sys
from statistics import mean

import put_paths
import put_stats

JUDGE_DIR = os.path.join(put_paths.HERE, "judge")
ANCHORS = os.path.join(JUDGE_DIR, "anchors.json")
REFERENCE = os.path.join(JUDGE_DIR, "reference-scores.json")
CRITERIA = ("convention_fit", "design_fit", "test_quality")


def _j(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select(label, n=8):
    summary = _j(os.path.join(put_paths.RUNS, f"{label}.json"))
    recs = [r for r in summary["records"] if r.get("verdict")]
    crits = ("correctness", "convention_fit", "design_fit", "test_quality", "graph_use")
    recs.sort(key=lambda r: (mean(r["verdict"][c] for c in crits), r["run"]))
    if len(recs) <= n:
        chosen = recs
    else:
        step = (len(recs) - 1) / (n - 1)
        idx = sorted({round(i * step) for i in range(n)})
        chosen = [recs[i] for i in idx]
    anchors = [{"id": f"A{i + 1}", "run": r["run"], "task": r["task"]} for i, r in enumerate(chosen)]
    os.makedirs(JUDGE_DIR, exist_ok=True)
    with open(ANCHORS, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"label": label, "anchors": anchors}, f, indent=1)
    return anchors


DEGRADED = os.path.join(JUDGE_DIR, "degraded")
FIXTURES = os.path.join(put_paths.CODEMAP, "remote", "tests", "fixtures", "put")
RUN_FILES = ("diff.patch", "calls.json", "tests.json", "answer.md", "meta.json", "fixture.json")


def anchor_dir(a):
    if a.get("source") == "fixture":
        return os.path.join(DEGRADED, a["id"])
    return os.path.join(put_paths.RUNS, *a["run"].replace("\\", "/").split("/"))


def extend(fixtures, model="opus", instance="backend-conventions"):
    """Add degraded variants (one broken rule each, training tasks only) as anchors and judge each twice.

    Baseline anchors carry only what the seed produced; when the seed is good, no anchor shows the judge a bad change
    and the calibration cannot say whether the judge would catch one."""
    import put_contract
    import put_judge
    doc = _j(ANCHORS)
    tasks = {t["id"]: t for t in put_contract.tasks(instance)}
    have = {a.get("fixture") for a in doc["anchors"]}
    for name in fixtures:
        if name in have:
            continue
        meta = _j(os.path.join(FIXTURES, name, "fixture.json"))
        task = tasks[meta["task"]]
        if task["split"] != "train":
            raise SystemExit(f"{name}: hold-out tasks enter nothing but certification")
        aid = f"A{len(doc['anchors']) + 1}"
        dst = os.path.join(DEGRADED, aid)
        os.makedirs(dst, exist_ok=True)
        for f in RUN_FILES:
            if os.path.exists(os.path.join(FIXTURES, name, f)):
                shutil.copy2(os.path.join(FIXTURES, name, f), dst)
        doc["anchors"].append({"id": aid, "source": "fixture", "fixture": name, "task": meta["task"],
                               "run": f"judge/degraded/{aid}", "breaks": meta["must_fail"]})
        with open(ANCHORS, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1)
    todo = [a for a in doc["anchors"] if a.get("source") == "fixture"]
    for a in todo:
        for vname in (put_judge.VERDICT, put_judge.VERDICT_REPEAT):
            put_judge.judge_run(anchor_dir(a), tasks[a["task"]], instance, model, name=vname)
    return todo


def sheet():
    """-> markdown per anchor: the task, what changed (files), the key added lines, the tests. No judge scores (blind)."""
    doc = _j(ANCHORS)
    out = []
    for a in doc["anchors"]:
        rd = anchor_dir(a)
        with open(os.path.join(rd, "diff.patch"), encoding="utf-8") as f:
            diff = f.read()
        tests = _j(os.path.join(rd, "tests.json"), {})
        out.append({"id": a["id"], "task": a["task"], "run": a["run"], "diff": diff, "tests": tests})
    return out


def _agreement(j, h):
    jb = [1 if x >= 4 else 0 for x in j]
    hb = [1 if x >= 4 else 0 for x in h]
    k = put_stats.kappa(jb, hb)
    return {"n": len(j), "exact": round(mean(1.0 if a == b else 0.0 for a, b in zip(j, h)), 3),
            "agreement": round(k["agreement"], 3), "prevalence": round(k["prevalence"], 3), "kappa": round(k["kappa"], 3),
            "ac1": round(put_stats.ac1(jb, hb), 3), "spearman": round(put_stats.spearman(j, h), 3),
            "mean_abs_diff": round(mean(abs(a - b) for a, b in zip(j, h)), 3)}


def compare(rows, criteria=CRITERIA):
    """rows: [(anchor id, source, judge verdict, judge repeat verdict, reference scores)] -> the calibration numbers.

    For each criterion and for all three pooled: exact agreement, agreement binarised at >= 4 with its prevalence,
    kappa and AC1, Spearman on the raw scores, mean absolute difference; and for every cell where the judge's first
    pass differs from the reference, whether its second pass lands on the reference (a gap inside the judge's own
    repeatability) or not (a gap that repeats: bias or blind spot)."""
    out = {"criteria": {}, "gaps": []}
    pool_j, pool_h = [], []
    for c in criteria:
        pairs = [(v.get(c), h.get(c)) for _, _, v, _, h in rows if v.get(c) is not None and h.get(c) is not None]
        if not pairs:
            continue
        j, h = [p[0] for p in pairs], [p[1] for p in pairs]
        out["criteria"][c] = _agreement(j, h) | {"judge": j, "reference": h}
        pool_j += j
        pool_h += h
    if pool_j:
        out["pooled"] = _agreement(pool_j, pool_h)
    for aid, source, v, vr, h in rows:
        for c in criteria:
            if v.get(c) is None or h.get(c) is None or v[c] == h[c]:
                continue
            out["gaps"].append({"anchor": aid, "source": source, "criterion": c, "judge": v[c], "judge_repeat": vr.get(c),
                                "reference": h[c], "crosses_line": (v[c] >= 4) != (h[c] >= 4),
                                "repeat_closes": vr.get(c) is not None and abs(vr[c] - h[c]) < abs(v[c] - h[c])})
    g = out["gaps"]
    out["gap_summary"] = {"cells": len(pool_j), "gaps": len(g), "crossing_the_line": sum(x["crosses_line"] for x in g),
                          "closed_by_the_repeat": sum(x["repeat_closes"] for x in g)}
    return out


def rejudged_dir(a, label):
    """The copy of an anchor's run under a rejudge label (put_cli rejudge): same layout, new verdicts."""
    if a.get("source") == "fixture":
        return os.path.join(put_paths.RUNS, label, "anchors", a["id"])
    parts = a["run"].replace("\\", "/").split("/")
    return os.path.join(put_paths.RUNS, label, *parts[1:])


def score(rubric="r4", label=None):
    """The judge's verdicts under `rubric` (read from the rejudge `label`'s copies when given) against the reference.
    -> judge/calibration-<rubric>.json"""
    doc, ref = _j(ANCHORS), _j(REFERENCE)
    rows = []
    for a in doc["anchors"]:
        h = (ref or {}).get("anchors", {}).get(a["id"])
        if not h:
            continue
        rd = rejudged_dir(a, label) if label else anchor_dir(a)
        v = (_j(os.path.join(rd, f"verdict-{rubric}.json"), {}) or {}).get("verdict") or {}
        vr = (_j(os.path.join(rd, f"verdict-{rubric}-repeat.json"), {}) or {}).get("verdict") or {}
        rows.append((a["id"], a.get("source", "baseline"), v, vr, h))
    result = {"label": label or doc["label"], "rubric": rubric, "n_anchors": len(rows),
              "by_source": {s: sum(1 for r in rows if r[1] == s) for s in sorted({r[1] for r in rows})}}
    result.update(compare(rows))
    result["per_anchor"] = [{"anchor": aid, "source": src, "judge": {c: v.get(c) for c in CRITERIA},
                             "judge_repeat": {c: vr.get(c) for c in CRITERIA}, "reference": {c: h.get(c) for c in CRITERIA}}
                            for aid, src, v, vr, h in rows]
    with open(os.path.join(JUDGE_DIR, f"calibration-{rubric}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "sheet", "extend", "score"])
    ap.add_argument("--label")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--fixtures", nargs="*", default=[])
    ap.add_argument("--rubric", default="r4")
    a = ap.parse_args(argv)
    if a.cmd == "select":
        print(json.dumps(select(a.label, a.n), indent=1))
    elif a.cmd == "sheet":
        for s in sheet():
            print(f"### {s['id']} — {s['task']}\n{s['tests']}\n{s['diff'][:3000]}\n")
    elif a.cmd == "extend":
        print(json.dumps(extend(a.fixtures), indent=1))
    else:
        print(json.dumps(score(a.rubric, a.label), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
