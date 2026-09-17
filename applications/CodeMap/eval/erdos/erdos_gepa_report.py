"""The report of one GEPA run over Erdős's manual.

    PYTHONUTF8=1 python eval/erdos/erdos_gepa_report.py --label 2026-09-17-gepa2      writes eval/erdos/runs/<label>.report.md

Reads the adapter's log (`runs/<label>/gepa.log.jsonl`: one line per graded run, refusal and reflection), each
candidate's manual body (`runs/<label>/<sha>/skill_body.md`) and, when the run finished, its summary
(`runs/<label>.json`). Writes, per candidate: the score on every problem, the mean of each judge criterion and of each
adherence check; the judge's noise on the seed; and what the best candidate changed in the manual, section by section,
with the diff against the seed.
"""

import argparse
import difflib
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CRITERIA = ("correctness", "completeness", "architecture_fit", "graph_use")
CHECKS = ("graph_pass", "key_files_read", "facts_grounded", "one_pass", "contract")
SECTION_RX = re.compile(r"^<([a-z_]+)(?: [^<>]*)?>$")


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return round(sum(values) / len(values), 2) if values else None


def _cell(v):
    return "" if v is None else str(v)


def read_log(path):
    evals, refused, reflections = defaultdict(dict), [], 0
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("event") == "eval":
                evals[rec["candidate"]][rec["problem"]] = rec          # the last grading of a run wins
            elif rec.get("event") == "refused":
                refused.append(rec)
            elif rec.get("event") == "reflect":
                reflections += 1
    return evals, refused, reflections


def sections(body):
    """-> {tag: text} for each top-level section line of the manual body (nested sections stay inside their parent)."""
    out, stack, current = {}, [], None
    for line in body.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        m = SECTION_RX.match(s)
        if m and not s.endswith("/>"):
            stack.append(m.group(1))
            if len(stack) == 2:
                current = m.group(1)
                out[current] = ""
        if current:
            out[current] += line + "\n"
        if re.match(r"^</([a-z_]+)>$", s) and stack:
            if len(stack) == 2:
                current = None
            stack.pop()
    return out


def build(label, runs_dir=None):
    runs_dir = runs_dir or os.path.join(HERE, "runs")
    run_dir = os.path.join(runs_dir, label)
    evals, refused, reflections = read_log(os.path.join(run_dir, "gepa.log.jsonl"))
    doc_path = os.path.join(runs_dir, f"{label}.json")
    doc = json.load(open(doc_path, encoding="utf-8")) if os.path.exists(doc_path) else {}
    order = [c["sha"] for c in doc.get("candidates", [])] or list(evals)
    for sha in evals:
        if sha not in order:
            order.append(sha)
    problems = sorted({pid for runs in evals.values() for pid in runs})
    val = {c["sha"]: c.get("val_score") for c in doc.get("candidates", [])}

    lines = [f"# GEPA over Erdős's manual — {label}", ""]
    if doc:
        lines += [f"Mission: {doc.get('mission')}.", "",
                  f"Score weights: " + ", ".join(f"{k} {w}" for k, w in (doc.get("weights") or {}).items()) + ". "
                  f"Models: Erdős `{doc.get('model')}`, judge `{doc.get('judge_model')}` (rubric {doc.get('judge_rubric')}), "
                  f"reflector `{doc.get('reflection_model')}`.", "",
                  f"Seed validation score {doc.get('seed_val_score')}; best candidate {doc.get('best_idx')} at {doc.get('best_val_score')}; "
                  f"improved: {doc.get('improved')}. Metric calls {doc.get('metric_calls')}, reflections {reflections}, "
                  f"{round((doc.get('seconds') or 0) / 60)} minutes. In-sample: the five problems are the training and the validation set.", ""]
        noise = doc.get("judge_noise") or {}
        if noise:
            lines += ["Judge noise on the seed (the same answers graded twice, mean absolute difference): " +
                      ", ".join(f"{k} {v}" for k, v in noise.items() if k != "problems") + f" over {noise.get('problems')} problems.", ""]
    else:
        lines += ["The run has not finished: no summary yet; the tables come from the log.", ""]

    lines += ["## Candidates", "", "| # | candidate | validation score | " + " | ".join(problems) + " | " + " | ".join(CRITERIA) +
              " | adherence |", "|" + "---|" * (4 + len(problems) + len(CRITERIA))]
    for i, sha in enumerate(order):
        runs = evals.get(sha, {})
        crit = [_mean([(r.get("scores") or {}).get(k) for r in runs.values()]) for k in CRITERIA]
        lines.append(f"| {i} | `{sha}` | {_cell(val.get(sha))} | " + " | ".join(_cell((runs.get(p) or {}).get("score")) for p in problems) +
                     " | " + " | ".join(_cell(c) for c in crit) + f" | {_cell(_mean([r.get('adherence') for r in runs.values()]))} |")
    lines += ["", "## Adherence checks (mean over the problems graded)", "", "| # | candidate | " + " | ".join(CHECKS) + " |",
              "|" + "---|" * (2 + len(CHECKS))]
    for i, sha in enumerate(order):
        runs = evals.get(sha, {})
        lines.append(f"| {i} | `{sha}` | " + " | ".join(_cell(_mean([(r.get("checks") or {}).get(k) for r in runs.values()]))
                                                       for k in CHECKS) + " |")
    if refused:
        lines += ["", f"## Refused candidates ({len(refused)})", ""]
        lines += [f"- `{r.get('candidate')}`: " + "; ".join(r.get("problems") or []) for r in refused]

    best_idx = doc.get("best_idx")
    if order and best_idx not in (None, 0) and best_idx < len(order):
        seed_path = os.path.join(run_dir, order[0], "skill_body.md")
        best_path = os.path.join(run_dir, order[best_idx], "skill_body.md")
        if os.path.exists(seed_path) and os.path.exists(best_path):
            seed, best = open(seed_path, encoding="utf-8").read(), open(best_path, encoding="utf-8").read()
            a, b = sections(seed), sections(best)
            lines += ["", f"## What candidate {best_idx} changed", "", "| section | seed lines | candidate lines | changed |", "|---|---|---|---|"]
            for tag in list(dict.fromkeys(list(a) + list(b))):
                lines.append(f"| {tag} | {a.get(tag, '').count(chr(10))} | {b.get(tag, '').count(chr(10))} | "
                             f"{'yes' if a.get(tag) != b.get(tag) else ''} |")
            diff = difflib.unified_diff(seed.splitlines(), best.splitlines(), "seed/SKILL.md body", f"candidate-{best_idx}/SKILL.md body", lineterm="")
            lines += ["", "```diff", *diff, "```"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    a = ap.parse_args(argv)
    text = build(a.label)
    with open(os.path.join(HERE, "runs", f"{a.label}.report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
