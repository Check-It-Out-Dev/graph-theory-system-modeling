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
import sys
from statistics import mean

import put_paths
import put_stats

JUDGE_DIR = os.path.join(put_paths.HERE, "judge")
ANCHORS = os.path.join(JUDGE_DIR, "anchors.json")
OWNER = os.path.join(JUDGE_DIR, "owner-scores.json")
OUT = os.path.join(JUDGE_DIR, "calibration.json")
OWNER_CRITERIA = ("convention_fit", "design_fit", "test_quality")


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


def sheet():
    """-> markdown per anchor: the task, what changed (files), the key added lines, the tests. No judge scores (blind)."""
    doc = _j(ANCHORS)
    out = []
    for a in doc["anchors"]:
        rd = os.path.join(put_paths.RUNS, *a["run"].replace("\\", "/").split("/"))
        with open(os.path.join(rd, "diff.patch"), encoding="utf-8") as f:
            diff = f.read()
        tests = _j(os.path.join(rd, "tests.json"), {})
        out.append({"id": a["id"], "task": a["task"], "run": a["run"], "diff": diff, "tests": tests})
    return out


def score():
    doc, owner = _j(ANCHORS), _j(OWNER)
    rows = []
    for a in doc["anchors"]:
        rd = os.path.join(put_paths.RUNS, *a["run"].replace("\\", "/").split("/"))
        v = (_j(os.path.join(rd, "verdict-r4.json"), {}) or {}).get("verdict") or {}
        rows.append((a["id"], v, (owner or {}).get(a["id"], {})))
    result = {"label": doc["label"], "n_anchors": len(rows), "criteria": {}}
    for c in OWNER_CRITERIA:
        pairs = [(v.get(c), o.get(c)) for _, v, o in rows if v.get(c) is not None and o.get(c) is not None]
        if not pairs:
            continue
        j = [p[0] for p in pairs]
        h = [p[1] for p in pairs]
        jb = [1 if x >= 4 else 0 for x in j]
        hb = [1 if x >= 4 else 0 for x in h]
        k = put_stats.kappa(jb, hb)
        result["criteria"][c] = {"n": len(pairs), "agreement": round(k["agreement"], 3), "prevalence": round(k["prevalence"], 3),
                                 "kappa": round(k["kappa"], 3), "ac1": round(put_stats.ac1(jb, hb), 3),
                                 "spearman": round(put_stats.spearman(j, h), 3),
                                 "mean_abs_diff": round(mean(abs(a - b) for a, b in pairs), 3),
                                 "judge": j, "owner": h}
    all_j = [x for c in result["criteria"].values() for x in c["judge"]]
    all_h = [x for c in result["criteria"].values() for x in c["owner"]]
    if all_j:
        jb = [1 if x >= 4 else 0 for x in all_j]
        hb = [1 if x >= 4 else 0 for x in all_h]
        k = put_stats.kappa(jb, hb)
        result["pooled"] = {"n": len(all_j), "agreement": round(k["agreement"], 3), "prevalence": round(k["prevalence"], 3),
                            "kappa": round(k["kappa"], 3), "ac1": round(put_stats.ac1(jb, hb), 3),
                            "spearman": round(put_stats.spearman(all_j, all_h), 3)}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "sheet", "score"])
    ap.add_argument("--label")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "select":
        print(json.dumps(select(a.label, a.n), indent=1))
    elif a.cmd == "sheet":
        for s in sheet():
            print(f"### {s['id']} — {s['task']}\n{s['tests']}\n{s['diff'][:3000]}\n")
    else:
        print(json.dumps(score(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
