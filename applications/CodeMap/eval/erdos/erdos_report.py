"""One table per label: the Erdős pairs side by side, solve and verify phases apart.

    PYTHONUTF8=1 python eval/erdos/erdos_report.py --label 2026-09-17      writes eval/erdos/runs/<label>.report.md

Per problem and arm: API calls, tool calls (graph / files), the bytes of tool results, the plain and the
price-weighted token sums, and the seconds, for the solve phase and the verify phase; then the judge's
overall score, the key's must_find recall and the unknown file names. The totals row sums the five
problems; the ratio row is erdos / general (below 1 means Erdős spent less).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _cell(v):
    return "" if v is None else (f"{v:,}" if isinstance(v, int) else (f"{v:,.1f}" if isinstance(v, float) else str(v)))


def build(runs, judge=None):
    judged = {j["problem"]: j for j in (judge or {}).get("problems", [])}
    lines = [f"# Erdős pairs — {runs['label']}", "",
             f"Model `{runs['meta'].get('model')}`, prompt `{runs['meta'].get('prompt_version')}`, pack `{runs['meta'].get('pack_version')}`.", "",
             "| problem | arm | phase | calls | graph tools | file tools | result KB | tokens | weighted | seconds | overall | recall | unknown files |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    totals = {}
    for r in sorted(runs["rows"], key=lambda r: (r["problem"], r["arm"])):
        j = judged.get(r["problem"]) or {}
        score = (((j.get("judge") or {}).get("scores") or {}).get(r["arm"]) or {}).get("overall")
        det = (j.get("deterministic") or {}).get(r["arm"]) or {}
        for phase in ("solve", "verify"):
            ph = r[phase]
            files = ph["tool_calls"] - ph["graph_tool_calls"]
            lines.append(f"| {r['problem']} | {r['arm']} | {phase} | {ph['calls']} | {ph['graph_tool_calls']} | {files} | "
                         f"{ph['result_bytes'] / 1000:.1f} | {_cell(ph['tokens_sum'])} | {_cell(ph['tokens_weighted'])} | {ph['seconds']} | "
                         f"{_cell(score) if phase == 'solve' else ''} | {_cell((det.get('must_find') or {}).get('recall')) if phase == 'solve' else ''} | "
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
        lines += ["", "| problem | better | equivalent | why |", "|---|---|---|---|"]
        for pid, j in sorted(judged.items()):
            v = j.get("judge") or {}
            lines.append(f"| {pid} | {v.get('better')} | {v.get('equivalent')} | {(v.get('why') or '').replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    a = ap.parse_args(argv)
    runs = json.load(open(os.path.join(HERE, "runs", f"{a.label}.json"), encoding="utf-8"))
    jp = os.path.join(HERE, "runs", f"{a.label}.judge.json")
    judge = json.load(open(jp, encoding="utf-8")) if os.path.exists(jp) else None
    text = build(runs, judge)
    with open(os.path.join(HERE, "runs", f"{a.label}.report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
