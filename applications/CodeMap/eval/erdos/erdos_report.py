"""One table per label: the Erdős pairs side by side, solve and verify phases apart.

    PYTHONUTF8=1 python eval/erdos/erdos_report.py --label 2026-09-17-v2 [--against 2026-09-17]
                                                   writes eval/erdos/runs/<label>.report.md

Per problem and arm: API calls, tool calls (graph / files), the bytes of tool results, the plain and the
price-weighted token sums, and the seconds, for the solve phase and the verify phase; then the judge's
overall score, the key's must_find recall and the unknown file names. The totals row sums the five
problems; the ratio row is erdos / general (below 1 means Erdős spent less).

`--against <label>` appends both runs side by side per arm: solve-phase cost, the earlier run's cost with its
verification phase included, and the judge's means. Each run's scores come from its own blind judge pass over
its own pairs, so the comparison inside a run is the measurement and the one across runs is context.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _cell(v):
    return "" if v is None else (f"{v:,}" if isinstance(v, int) else (f"{v:,.1f}" if isinstance(v, float) else str(v)))


def _recall(det):
    r = (det.get("must_find") or {}).get("recall")
    return "" if r is None else f"{r:.2f}"


JUDGE_KEYS = ("overall", "correctness", "design", "plan", "architecture_fit", "must_find_hits", "patterns_followed",
              "key_facts_supported", "gaps_found", "red_flags_made")


def arm_totals(runs):
    """-> {(arm, phase): sums} for the phases solve, verify and total."""
    totals = {}
    for r in runs["rows"]:
        for phase in ("solve", "verify", "total"):
            ph = r.get(phase)
            if not ph:
                continue
            t = totals.setdefault((r["arm"], phase), {"calls": 0, "tokens": 0, "weighted": 0.0, "seconds": 0.0, "graph": 0, "files": 0})
            t["calls"] += ph["calls"]; t["tokens"] += ph["tokens_sum"]; t["weighted"] += ph["tokens_weighted"]
            t["seconds"] += ph["seconds"]; t["graph"] += ph["graph_tool_calls"]; t["files"] += ph["tool_calls"] - ph["graph_tool_calls"]
    return totals


def judge_means(judge):
    """-> {arm: {key: mean over problems}} plus the mean must_find recall of the deterministic check."""
    out = {}
    problems = (judge or {}).get("problems", [])
    for arm in ("general", "erdos"):
        rows = [((j.get("judge") or {}).get("scores") or {}).get(arm) or {} for j in problems]
        recalls = [((j.get("deterministic") or {}).get(arm) or {}).get("must_find", {}).get("recall") for j in problems]
        recalls = [x for x in recalls if x is not None]
        if rows:
            m = {k: round(sum(r.get(k) or 0 for r in rows) / len(rows), 2) for k in JUDGE_KEYS}
            m["recall"] = round(sum(recalls) / len(recalls), 2) if recalls else None
            out[arm] = m
    return out


def comparison(runs, judge, earlier, earlier_judge):
    lines = ["", f"## Against {earlier['label']}", "",
             f"Prompts: `{earlier['meta'].get('prompt_version')}` then `{runs['meta'].get('prompt_version')}`. Each run was judged in its own "
             "blind pass over its own pairs: read the erdos / general comparison inside a run first.", "",
             "| run | arm | phases | calls | graph tools | file tools | tokens | weighted | seconds | overall | correctness | architecture fit | red flags | recall |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label_runs, label_judge, phases in ((earlier, earlier_judge, ("solve", "total")), (runs, judge, ("solve",))):
        totals, means = arm_totals(label_runs), judge_means(label_judge)
        for arm in ("general", "erdos"):
            for phase in phases:
                t = totals.get((arm, phase))
                if not t:
                    continue
                m = means.get(arm) or {}
                name = "solve" if phase == "solve" else "solve + verify"
                lines.append(f"| {label_runs['label']} | {arm} | {name} | {t['calls']} | {t['graph']} | {t['files']} | {t['tokens']:,} | "
                             f"{t['weighted']:,.0f} | {t['seconds']:.0f} | {_cell(m.get('overall'))} | {_cell(m.get('correctness'))} | {_cell(m.get('architecture_fit'))} | "
                             f"{_cell(m.get('red_flags_made'))} | {'' if m.get('recall') is None else format(m['recall'], '.2f')} |")
    return lines


def build(runs, judge=None, against=None):
    judged = {j["problem"]: j for j in (judge or {}).get("problems", [])}
    lines = [f"# Erdős pairs — {runs['label']}", "",
             f"Model `{runs['meta'].get('model')}`, prompt `{runs['meta'].get('prompt_version')}`, pack `{runs['meta'].get('pack_version')}`, "
             f"judge rubric `{(judge or {}).get('rubric', 'r1')}`." + (f" General runs reused from `{runs['meta']['general_reused_from']}`."
                                                                      if runs['meta'].get('general_reused_from') else ""), "",
             "| problem | arm | phase | calls | graph tools | graph before files | file tools | result KB | tokens | weighted | seconds | overall | architecture fit | recall | unknown files |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    totals = {}
    for r in sorted(runs["rows"], key=lambda r: (r["problem"], r["arm"])):
        j = judged.get(r["problem"]) or {}
        scores = ((j.get("judge") or {}).get("scores") or {}).get(r["arm"]) or {}
        score, fit = scores.get("overall"), scores.get("architecture_fit")
        det = (j.get("deterministic") or {}).get(r["arm"]) or {}
        for phase in ("solve", "verify"):
            ph = r[phase]
            files = ph["tool_calls"] - ph["graph_tool_calls"]
            lines.append(f"| {r['problem']} | {r['arm']} | {phase} | {ph['calls']} | {ph['graph_tool_calls']} | {_cell(ph.get('graph_before_files'))} | {files} | "
                         f"{ph['result_bytes'] / 1000:.1f} | {_cell(ph['tokens_sum'])} | {_cell(ph['tokens_weighted'])} | {ph['seconds']} | "
                         f"{_cell(score) if phase == 'solve' else ''} | {_cell(fit) if phase == 'solve' else ''} | {_recall(det) if phase == 'solve' else ''} | "
                         f"{len((det.get('files') or {}).get('unknown') or []) if phase == 'solve' and det else ''} |")
            t = totals.setdefault((r["arm"], phase), {"calls": 0, "tokens": 0, "weighted": 0.0, "seconds": 0.0, "graph": 0, "files": 0})
            t["calls"] += ph["calls"]; t["tokens"] += ph["tokens_sum"]; t["weighted"] += ph["tokens_weighted"]
            t["seconds"] += ph["seconds"]; t["graph"] += ph["graph_tool_calls"]; t["files"] += files
    lines += ["", "| arm | phase | calls | graph tools | file tools | tokens | weighted | seconds |", "|---|---|---|---|---|---|---|---|"]
    for (arm, phase), t in sorted(totals.items()):
        lines.append(f"| {arm} | {phase} | {t['calls']} | {t['graph']} | {t['files']} | {t['tokens']:,} | {t['weighted']:,.0f} | {t['seconds']:.0f} |")
    for phase in ("solve", "verify"):
        g, e = totals.get(("general", phase)), totals.get(("erdos", phase))
        if g and e:
            ratio = lambda k: round(e[k] / g[k], 2) if g[k] else None
            lines.append(f"| erdos / general | {phase} | {ratio('calls')} | | | {ratio('tokens')} | {ratio('weighted')} | {ratio('seconds')} |")
    if judged:
        lines += ["", "The judge saw the answers as A and B; its reasons use those letters.", "",
                  "| problem | A was | better | equivalent | why |", "|---|---|---|---|---|"]
        for pid, j in sorted(judged.items()):
            v = j.get("judge") or {}
            lines.append(f"| {pid} | {(j.get('blind_order') or {}).get('A')} | {v.get('better')} | {v.get('equivalent')} | "
                         f"{(v.get('why') or '').replace('|', '/')} |")
        keys = JUDGE_KEYS
        lines += ["", "| arm (mean over problems) | overall | correctness | design | plan | architecture fit | must_find hits | patterns followed | key facts | gaps found | red flags |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ("general", "erdos"):
            rows = [((j.get("judge") or {}).get("scores") or {}).get(arm) or {} for j in judged.values()]
            if rows:
                m = [round(sum(r.get(k) or 0 for r in rows) / len(rows), 2) for k in keys]
                lines.append(f"| {arm} | " + " | ".join(str(x) for x in m) + " |")
    if against:
        lines += comparison(runs, judge, *against)
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--against", default=None)
    a = ap.parse_args(argv)

    def load(label):
        runs = json.load(open(os.path.join(HERE, "runs", f"{label}.json"), encoding="utf-8"))
        for name in (f"{label}.judge-r2.json", f"{label}.judge.json"):   # the newest rubric a label was judged with
            jp = os.path.join(HERE, "runs", name)
            if os.path.exists(jp):
                return runs, json.load(open(jp, encoding="utf-8"))
        return runs, None

    runs, judge = load(a.label)
    text = build(runs, judge, against=load(a.against) if a.against else None)
    with open(os.path.join(HERE, "runs", f"{a.label}.report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
