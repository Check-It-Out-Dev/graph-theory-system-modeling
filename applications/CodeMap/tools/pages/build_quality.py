"""The public quality page: every committed night, decision and optimiser run, as one static HTML.

    python tools/pages/build_quality.py --out site/quality

No model, no network, no dependency: it reads the artifacts this repository commits —
`eval/quality/runs/<date>.json` (the nightly rates), `eval/judge/runs/<date>.json` (calibration),
`eval/humans/runs/<date>.summary.json` (the personas' nights), `graph/ledger/*.json` + `*.drift.json`
(decisions and drift), `eval/optimize/runs/*.json` (prompt optimisation), `prompts/navigator/PROMPT_LOG.md`,
`observability/grafana/public-urls.md` (the live dashboards) and `docs/proofs/*.jpg` (screenshots) —
and writes `index.html` (inline CSS + SVG, phone-wide, dark and light) plus `data.json` (the same
numbers, for anyone who wants to draw their own). Published by `nightly.yml` to GitHub Pages under
`/quality/`; the page is the telemetry's public face, the dashboards are its live one.
"""

import argparse
import glob
import html
import json
import os
import re
import sys
import time

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_URL = "https://github.com/Check-It-Out-Dev/graph-theory-system-modeling"
MCP_URL = "https://codemap.checkitout.app/mcp"


def jload(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect(root=R):
    nights = []
    for p in sorted(glob.glob(os.path.join(root, "eval", "quality", "runs", "*.json"))):
        d = jload(p)
        if not d or "date" not in d:
            continue
        date = d["date"]
        judge = jload(os.path.join(root, "eval", "judge", "runs", f"{date}.json")) or {}
        humans = jload(os.path.join(root, "eval", "humans", "runs", f"{date}.summary.json")) or {}
        kappa = d.get("codemap_judge_kappa") or {}
        gain = d.get("codemap_gain") or {}
        nights.append({
            "date": date, "n": d.get("n", {}),
            "grounded": d.get("codemap_grounded_rate"), "correct": d.get("codemap_correct_rate"), "helpful": d.get("codemap_helpful_rate"),
            "located": d.get("codemap_located_rate"), "abstention": (d.get("codemap_abstention_rate") or {}),
            "pointer_verified": d.get("codemap_pointer_verified_rate"), "rating_mean": d.get("codemap_rating_mean"),
            "rating_ge4": d.get("codemap_rating_ge4_rate"), "credits_per_correct": d.get("codemap_credits_per_correct_answer"),
            "credits_total": d.get("codemap_credits_total") or {}, "tokens": d.get("codemap_tokens_total") or {},
            "requests": d.get("codemap_requests_total") or {}, "kappa": kappa, "disputes": d.get("codemap_disputes_total"),
            "drift_rate": d.get("codemap_version_drift_rate"), "drift": d.get("codemap_version_drift") or {},
            "coverage": d.get("codemap_graph_coverage_ratio") or {}, "gain": {k: gain.get(k) for k in ("n_pairs", "tokens_ratio_mean", "turns_delta_mean", "seconds_delta_mean")},
            "cache_read_ratio": d.get("codemap_cache_read_ratio"), "latency_p95": d.get("codemap_latency_ms_p95"),
            "judge_usage": ((judge.get("summary") or {}).get("usage") or {}), "judge_n": (judge.get("summary") or {}).get("n"),
            "personas": {k: {"conversations": v.get("conversations"), "credits": v.get("credits_spent"), "rating_mean": v.get("rating_mean"),
                             "turns": v.get("turns"), "misses": v.get("misses")} for k, v in (humans.get("by_persona") or {}).items()},
            "partial": humans.get("partial"),
        })
    decisions = []
    for p in sorted(glob.glob(os.path.join(root, "graph", "ledger", "*.json"))):
        name = os.path.basename(p)
        if ".drift." in name or ".reclue." in name:
            continue
        d = jload(p)
        if not d or "pack_version" not in d:
            continue
        drift = jload(p.replace(".json", ".drift.json")) or {}
        reclue = jload(p.replace(".json", ".reclue.json")) or {}
        decisions.append({"version": d["pack_version"], "repo": d.get("repo"), "head": (d.get("head") or "")[:7], "by": d.get("decided_by"),
                          "at": d.get("decided_at"), "kind": (d.get("command") or {}).get("kind"), "assignments": len(d.get("assignments") or []),
                          "new_subsystems": [n.get("name") for n in d.get("new_subsystems") or []], "changed": d.get("changed_subsystems") or [],
                          "invalidated": len(d.get("mfq_invalidated") or []), "rejected": d.get("rejected"),
                          "drift_rate": drift.get("drift_rate"), "drifted": drift.get("drifted"), "compared": drift.get("compared"),
                          "coverage": (d.get("coverage") or {}).get("ratio"),
                          "reclued": [r["sub_id"] for r in (reclue.get("results") or []) if r.get("status") == "reclued"]})
    runs = []
    for p in sorted(glob.glob(os.path.join(root, "eval", "optimize", "runs", "*.json"))):
        d = jload(p)
        if not d or "mode" not in d:
            continue
        runs.append({"date": d.get("date"), "mode": d.get("mode"), "seed": d.get("seed_val_score"), "best": d.get("best_val_score"),
                     "candidates": d.get("candidates"), "metric_calls": d.get("metric_calls"), "win": d.get("win"), "seconds": d.get("seconds"),
                     "where": "modal" if "modal" in (d.get("date") or "") else "box"})
    prompts = []
    log = os.path.join(root, "prompts", "navigator", "PROMPT_LOG.md")
    if os.path.exists(log):
        for line in open(log, encoding="utf-8"):
            if re.match(r"^\| v\d+ ", line):  # rows, never the header
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 6:
                    prompts.append({"version": cells[0], "built": cells[1], "parent": cells[2], "why": cells[4], "gate": cells[5]})
    dashboards = []
    urls = os.path.join(root, "observability", "grafana", "public-urls.md")
    if os.path.exists(urls):
        for line in open(urls, encoding="utf-8"):
            m = re.match(r"\|\s*(CodeMap[^|]+)\|\s*`([^`]+)`\s*\|\s*(https://\S+)\s*\|", line)
            if m:
                dashboards.append({"title": m.group(1).strip(), "uid": m.group(2), "url": m.group(3)})
    proofs = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(root, "docs", "proofs", "*.jpg")))]
    return {"schema": 1, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "nights": nights, "decisions": decisions,
            "optimizer_runs": runs, "prompts": prompts, "dashboards": dashboards, "proofs": proofs, "mcp_url": MCP_URL, "repo": REPO_URL}


# ----------------------------------------------------------------------------- rendering

def scalar(x, prefer=("nav-sonnet", "all", "oracle")):
    """A number out of a per-key dict (the navigator tier first), or the value itself."""
    if isinstance(x, dict):
        for k in prefer:
            if isinstance(x.get(k), (int, float)):
                return x[k]
        vals = [v for v in x.values() if isinstance(v, (int, float))]
        return max(vals) if vals else None
    return x


def pct(x):
    x = scalar(x)
    return "—" if not isinstance(x, (int, float)) else f"{x * 100:.0f} %"


def num(x, d=1):
    x = scalar(x)
    if not isinstance(x, (int, float)):
        return "—"
    return f"{x:,.{d}f}" if isinstance(x, float) else f"{x:,}"


def esc(s):
    return html.escape(str(s if s is not None else ""))


def sparkline(values, w=220, h=48, unit_pct=True):
    """Inline SVG line with points; a single night draws one point, honestly."""
    pts = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
    if not pts:
        return '<svg class="spark" viewBox="0 0 %d %d" role="img" aria-label="no data"></svg>' % (w, h)
    lo, hi = (0.0, 1.0) if unit_pct else (min(v for _, v in pts), max(v for _, v in pts))
    if hi == lo:
        hi = lo + 1
    n = max(1, len(values) - 1)
    xy = [(6 + (w - 12) * i / n, h - 6 - (h - 12) * (v - lo) / (hi - lo)) for i, v in pts]
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in xy)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3"/>' for x, y in xy)
    return f'<svg class="spark" viewBox="0 0 {w} {h}" role="img"><path d="{path}"/>{dots}</svg>'


def bar(label, value, maximum=1.0):
    v = 0 if value is None else max(0.0, min(1.0, value / maximum))
    return (f'<div class="bar"><span class="bar-label">{esc(label)}</span><span class="bar-track"><span class="bar-fill" style="width:{v * 100:.1f}%"></span></span>'
            f'<span class="bar-value">{pct(value) if maximum == 1.0 else num(value)}</span></div>')


CSS = """
:root{--bg:#0e1116;--card:#161b22;--ink:#e6edf3;--muted:#9da7b3;--line:#2b3441;--accent:#3fb950;--accent2:#d29922;--accent3:#58a6ff;--danger:#f85149}
@media (prefers-color-scheme: light){:root{--bg:#f6f8fa;--card:#ffffff;--ink:#1f2328;--muted:#59636e;--line:#d0d7de}}
*{box-sizing:border-box}body{margin:0;padding:24px 16px 48px;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1120px;margin-inline:auto}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}h3{font-size:15px;margin:0 0 8px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
p{max-width:78ch}.muted{color:var(--muted)}a{color:var(--accent3)}code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;background:var(--card);padding:1px 5px;border-radius:4px;border:1px solid var(--line)}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.big{font-size:34px;font-weight:700;line-height:1.1;margin:2px 0 4px}.big.ok{color:var(--accent)}.big.warn{color:var(--accent2)}.big.info{color:var(--accent3)}
.spark{width:100%;height:48px;margin-top:6px}.spark path{fill:none;stroke:var(--accent3);stroke-width:2}.spark circle{fill:var(--accent3)}
.bar{display:grid;grid-template-columns:150px 1fr 60px;gap:10px;align-items:center;margin:6px 0}.bar-track{height:10px;background:var(--line);border-radius:5px;overflow:hidden}.bar-fill{display:block;height:100%;background:var(--accent)}.bar-value{text-align:right;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:14px}.tablewrap{overflow-x:auto}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;border:1px solid var(--line);font-size:12px;color:var(--muted)}.tag.ok{color:var(--accent);border-color:var(--accent)}.tag.no{color:var(--danger);border-color:var(--danger)}
.proofs{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}.proofs img{width:100%;height:auto;border:1px solid var(--line);border-radius:8px}figcaption{font-size:13px;color:var(--muted);margin-top:4px}
footer{margin-top:40px;color:var(--muted);font-size:13px}
"""


def render(data):
    nights = data["nights"]
    last = nights[-1] if nights else {}
    series = lambda k: [n.get(k) for n in nights]  # noqa: E731
    out = []
    out.append(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
               f"<title>CodeMap Remote — quality</title><style>{CSS}</style></head><body>")
    out.append("<h1>CodeMap Remote — measuring prompt quality in an AI system</h1>")
    out.append(f"<p class='muted'>Every number on this page is recomputed from an artifact committed in <a href='{REPO_URL}'>the repository</a>: "
               f"nightly quality runs, judge calibrations, the personas' nights, partition decisions with their drift, and prompt-optimisation runs. "
               f"No model runs to build it. Built {esc(data['built_at'])}. The live counterpart is the set of public Grafana Cloud dashboards below; "
               f"the served MCP is <code>{esc(MCP_URL)}</code>.</p>")
    # headline
    out.append("<h2>The latest night" + (f" — {esc(last.get('date'))}" if last else "") + "</h2>")
    if last:
        n = last.get("n") or {}
        kappa = (last.get("kappa") or {}).get("oracle")
        cards = [
            ("Grounded answers", pct(last.get("grounded")), "ok", "judge ≥ 4 on grounding, share of judged answers"),
            ("Correct answers", pct(last.get("correct")), "ok", "judge ≥ 4 on correctness"),
            ("Mean rating", num(last.get("rating_mean")), "info", "personas rate after verifying a pointer; unverified ratings cap at 3"),
            ("Judge κ vs oracle", num(kappa, 2), "ok" if (kappa or 0) >= 0.6 else "warn", "Cohen's κ of the judge's 'located' against the execution oracle (gate 0.6)"),
            ("Credits per correct answer", num(last.get("credits_per_correct")), "info", "credits are relative units, never currency"),
            ("Version drift", pct(last.get("drift_rate")), "warn" if (last.get("drift_rate") or 0) > 0.05 else "ok", "bank rows answering differently after the last pack decision, not invalidated on purpose"),
            ("Answers judged", num(n.get("judged")), "info", f"{num(n.get('asks'))} asks, {num(n.get('misses'))} misses reported, {num(n.get('refusals'))} budget refusals"),
            ("Honest abstention", pct((last.get("abstention") or {}).get("honest")), "ok", f"all {pct((last.get('abstention') or {}).get('all'))}, false {pct((last.get('abstention') or {}).get('false'))}"),
        ]
        out.append("<div class='grid'>")
        for title, value, cls, note in cards:
            out.append(f"<div class='card'><h3>{esc(title)}</h3><div class='big {cls}'>{esc(value)}</div><div class='muted'>{esc(note)}</div></div>")
        out.append("</div>")
    # trends
    out.append("<h2>Night over night</h2>")
    if len(nights) < 2:
        out.append("<p class='muted'>One night so far: a point, not a trend. Sparklines fill in as nights accumulate.</p>")
    out.append("<div class='grid'>")
    for title, key, unit in (("Grounded", "grounded", True), ("Correct", "correct", True), ("Mean rating (of 5)", "rating_mean", False),
                             ("Pointer-verified ratings", "pointer_verified", True), ("Version drift", "drift_rate", True), ("Credits per correct answer", "credits_per_correct", False)):
        vals = series(key)
        latest = vals[-1] if vals else None
        out.append(f"<div class='card'><h3>{esc(title)}</h3><div class='big info'>{pct(latest) if unit else num(latest)}</div>{sparkline(vals, unit_pct=unit)}"
                   f"<div class='muted'>{len([v for v in vals if v is not None])} night(s)</div></div>")
    out.append("</div>")
    # personas
    if last.get("personas"):
        out.append("<h2>Who asked — the synthetic users</h2>")
        out.append("<div class='tablewrap'><table><tr><th>persona</th><th class='n'>conversations</th><th class='n'>turns</th><th class='n'>credits</th><th class='n'>mean rating</th><th class='n'>misses</th></tr>")
        for pid, v in sorted(last["personas"].items()):
            out.append(f"<tr><td><code>{esc(pid)}</code></td><td class='n'>{num(v.get('conversations'))}</td><td class='n'>{num(v.get('turns'))}</td>"
                       f"<td class='n'>{num(v.get('credits'))}</td><td class='n'>{num(v.get('rating_mean'))}</td><td class='n'>{num(v.get('misses'))}</td></tr>")
        out.append("</table></div>")
        if last.get("partial"):
            out.append("<p class='muted'>This night ended early on a subscription rate limit and is recorded as partial.</p>")
    # cost
    if last.get("credits_total") or last.get("tokens"):
        out.append("<h2>What it cost — in credits and tokens, never currency</h2><div class='grid'>")
        ct = last.get("credits_total") or {}
        tot = sum(v for v in ct.values() if isinstance(v, (int, float))) or 1
        out.append("<div class='card'><h3>Credits by user</h3>")
        for k, v in sorted(ct.items(), key=lambda kv: -kv[1]):
            out.append(bar(k, v, maximum=tot))
        out.append("</div>")
        tk = last.get("tokens") or {}
        out.append("<div class='card'><h3>Tokens</h3>")
        for k in ("cached", "cache_creation", "completion", "prompt"):
            if k in tk:
                out.append(f"<div class='bar'><span class='bar-label'>{esc(k)}</span><span></span><span class='bar-value'>{num(tk[k])}</span></div>")
        out.append(f"<div class='muted'>cache-read ratio {pct(last.get('cache_read_ratio'))} · p95 latency {num(last.get('latency_p95'))} ms</div></div>")
        g = last.get("gain") or {}
        out.append(f"<div class='card'><h3>Gains vs no-CodeMap baselines</h3><div class='big info'>{num(g.get('n_pairs'))} pairs</div>"
                   f"<div class='muted'>tokens ratio {num(g.get('tokens_ratio_mean'), 2)} · turns Δ {num(g.get('turns_delta_mean'))} · seconds Δ {num(g.get('seconds_delta_mean'))}; "
                   f"a gain without its baseline row is ungated</div></div>")
        out.append("</div>")
    # decisions
    out.append("<h2>The graph follows the code — decisions on the pull request</h2>")
    if data["decisions"]:
        out.append("<div class='tablewrap'><table><tr><th>pack</th><th>repo @ head</th><th>decision</th><th>by</th><th class='n'>placed</th><th>new subsystems</th><th class='n'>FAQ invalidated</th><th class='n'>drift</th><th>reclued</th></tr>")
        for d in data["decisions"]:
            drift = f"{d['drifted']}/{d['compared']} ({pct(d['drift_rate'])})" if d.get("compared") else "—"
            out.append(f"<tr><td><code>{esc(d['version'])}</code></td><td>{esc(d['repo'])} @ <code>{esc(d['head'])}</code></td>"
                       f"<td><span class='tag {'no' if d.get('rejected') else 'ok'}'>{esc(d.get('kind'))}</span></td><td>{esc(d.get('by'))}</td>"
                       f"<td class='n'>{num(d['assignments'])}</td><td>{esc(', '.join(d['new_subsystems']) or '—')}</td><td class='n'>{num(d['invalidated'])}</td>"
                       f"<td class='n'>{esc(drift)}</td><td>{esc(', '.join(d['reclued']) or '—')}</td></tr>")
        out.append("</table></div>")
    else:
        out.append("<p class='muted'>No decision recorded yet.</p>")
    # optimiser
    out.append("<h2>The prompt improves on request — GEPA runs</h2>")
    if data["optimizer_runs"]:
        out.append("<div class='tablewrap'><table><tr><th>run</th><th>mode</th><th>where</th><th class='n'>seed (val)</th><th class='n'>best (val)</th><th class='n'>candidates</th><th class='n'>metric calls</th><th>outcome</th></tr>")
        for r in data["optimizer_runs"]:
            outcome = "promoted candidate" if r.get("win") else ("no win — recorded" if r.get("mode") == "gepa" else "seed measured")
            out.append(f"<tr><td><code>{esc(r['date'])}</code></td><td>{esc(r['mode'])}</td><td>{esc(r['where'])}</td><td class='n'>{num(r.get('seed'), 2)}</td>"
                       f"<td class='n'>{num(r.get('best'), 2)}</td><td class='n'>{num(r.get('candidates'))}</td><td class='n'>{num(r.get('metric_calls'))}</td>"
                       f"<td><span class='tag {'ok' if r.get('win') else ''}'>{esc(outcome)}</span></td></tr>")
        out.append("</table></div>")
    out.append("<div class='tablewrap'><table><tr><th>prompt</th><th>built</th><th>parent</th><th>why</th><th>gate</th></tr>")
    for p in data["prompts"]:
        out.append(f"<tr><td><code>{esc(p['version'])}</code></td><td>{esc(p['built'])}</td><td>{esc(p['parent'])}</td><td>{esc(p['why'])}</td><td>{esc(p['gate'])}</td></tr>")
    out.append("</table></div>")
    # dashboards + proofs
    out.append("<h2>Live dashboards (Grafana Cloud, public)</h2><ul>")
    for d in data["dashboards"]:
        out.append(f"<li><a href='{esc(d['url'])}'>{esc(d['title'])}</a> <span class='muted'>{esc(d['uid'])}</span></li>")
    out.append("</ul>")
    if data["proofs"]:
        out.append("<h2>Visual proofs</h2><div class='proofs'>")
        for name in data["proofs"]:
            cap = name.replace(".jpg", "").replace("-", " ")
            out.append(f"<figure><img src='proofs/{esc(name)}' alt='{esc(cap)}' loading='lazy'><figcaption>{esc(cap)}</figcaption></figure>")
        out.append("</div>")
    out.append(f"<footer>Security is designed to serve a small team, even behind a firewall — identity is a header from an enum and one shared token; "
               f"this deployment exists to show how to measure prompt quality in an AI system and how to observe it. Cards: "
               f"<a href='{REPO_URL}/blob/main/applications/CodeMap/MODEL_CARD.md'>model</a> · <a href='{REPO_URL}/blob/main/applications/CodeMap/EVAL_CARD.md'>evaluation</a> · "
               f"<a href='{REPO_URL}/blob/main/applications/CodeMap/DATA_CARD.md'>data</a> · <a href='{REPO_URL}/blob/main/applications/CodeMap/THREAT_MODEL.md'>threat model</a> · "
               f"<a href='{REPO_URL}/blob/main/applications/CodeMap/INCIDENTS.md'>incidents</a>. You are reading about an AI system; its answers can be wrong.</footer>")
    out.append("</body></html>")
    return "\n".join(out)


def build(out_dir, root=R):
    data = collect(root)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(render(data))
    with open(os.path.join(out_dir, "data.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    proofs_dir = os.path.join(out_dir, "proofs")
    os.makedirs(proofs_dir, exist_ok=True)
    import shutil
    for name in data["proofs"]:
        shutil.copyfile(os.path.join(root, "docs", "proofs", name), os.path.join(proofs_dir, name))
    return data


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    data = build(a.out)
    print(f"quality page: {len(data['nights'])} nights, {len(data['decisions'])} decisions, {len(data['optimizer_runs'])} optimizer runs, "
          f"{len(data['proofs'])} proofs → {a.out}/index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
