"""Quality rates: fold events + judge run + humans run into one artifact whose keys are the metric
names the dashboards use. Every number here has one home (this file's output); claims in prose are
rows in claims.json re-derived from it; the CI gate replays the committed fixture.

    python eval/quality/quality.py --date D --events <events.jsonl> [--judge eval/judge/runs/D.json]
                                   [--humans eval/humans/runs/D.jsonl] --out eval/quality/runs/D.json [--push]
"""

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(1.0 if x else 0.0 for x in xs) / len(xs), 4) if xs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _p95(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = max(0, int(round(0.95 * (len(xs) - 1))))
    return xs[k]


def load_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def compute(date, events, judge_doc=None, humans_rows=None):
    asks = [e for e in events if e.get("event_type") == "ask"]
    nav = [e for e in asks if str(e.get("tier", "")).startswith("nav-")]
    fb = [e for e in events if e.get("event_type") == "feedback"]
    misses = [e for e in events if e.get("event_type") == "miss"]
    refusals = [e for e in events if e.get("event_type") == "budget_refusal"]
    jrows = (judge_doc or {}).get("rows", [])
    judged = [r for r in jrows if r.get("judge")]
    cal = (judge_doc or {}).get("calibration", {})
    out = {"schema": 1, "date": date, "n": {"asks": len(asks), "navigator_answers": len(nav), "feedback": len(fb),
                                            "judged": len(judged), "misses": len(misses), "refusals": len(refusals)}}
    # --- quality (judge)
    out["codemap_located_rate"] = _rate([r["judge"].get("located", r["judge"]["correct"]) >= 4 for r in judged])
    out["codemap_grounded_rate"] = _rate([r["judge"]["grounded"] >= 4 for r in judged])
    out["codemap_correct_rate"] = _rate([r["judge"]["correct"] >= 4 for r in judged])
    out["codemap_helpful_rate"] = _rate([r["judge"]["helpful"] >= 4 for r in judged])
    abst = [r for r in judged if r.get("terminal") == "abstain"]
    out["codemap_abstention_rate"] = {"all": _rate([e.get("terminal") == "abstain" for e in nav]),
                                      "honest": _rate([r["judge"]["abstain"] >= 4 for r in abst]) if abst else None,
                                      "false": _rate([r["judge"]["abstain"] < 4 for r in abst]) if abst else None}
    out["codemap_oracle_success_rate"] = _rate([r["oracle"]["success"] for r in jrows if r.get("oracle", {}).get("has")])
    out["codemap_judge_kappa"] = {"oracle": cal.get("kappa_oracle"), "human": cal.get("kappa_human"),
                                  "anchor_now": (cal.get("anchors") or {}).get("kappa_now")}
    out["codemap_judge_agreement"] = cal.get("agreement_rr")
    out["codemap_judge_calibrated"] = cal.get("calibrated")
    out["codemap_disputes_total"] = sum(1 for r in jrows if r.get("disputed"))
    # --- happiness (users)
    ratings = [e.get("rating") for e in fb if e.get("rating") is not None]
    out["codemap_rating_mean"] = _mean(ratings)
    out["codemap_rating_ge4_rate"] = _rate([r >= 4 for r in ratings])
    out["codemap_pointer_verified_rate"] = _rate([bool(e.get("verified")) for e in fb])
    tags = {}
    for e in fb:
        for t in e.get("tags") or []:
            tags[t] = tags.get(t, 0) + 1
    out["codemap_feedback_tags_total"] = tags
    # --- usage
    by_tier = {}
    for e in asks:
        by_tier[e.get("tier")] = by_tier.get(e.get("tier"), 0) + 1
    out["codemap_requests_total"] = by_tier
    tok = {"prompt": 0, "completion": 0, "cached": 0, "cache_creation": 0}
    for e in nav:
        for k in tok:
            tok[k] += int((e.get("tokens") or {}).get(k, 0) or 0)
    out["codemap_tokens_total"] = tok
    denom = tok["cached"] + tok["cache_creation"] + tok["prompt"]
    out["codemap_cache_read_ratio"] = round(tok["cached"] / denom, 4) if denom else None
    out["codemap_steps_mean"] = _mean([e.get("steps") for e in nav])
    out["codemap_latency_ms_p95"] = {t: _p95([e.get("duration_ms") for e in asks if e.get("tier") == t]) for t in by_tier}
    # --- credits
    credits_by_user = {}
    for e in events:
        c = float(e.get("credits") or 0)
        if c:
            credits_by_user[e.get("user")] = round(credits_by_user.get(e.get("user"), 0.0) + c, 4)
    out["codemap_credits_total"] = credits_by_user
    total_credits = round(sum(credits_by_user.values()), 4)
    correct = sum(1 for r in judged if r["judge"]["correct"] >= 4)
    out["codemap_credits_per_correct_answer"] = round(total_credits / correct, 4) if correct else None
    out["codemap_credits_per_answer"] = round(total_credits / len(nav), 4) if nav else None
    out["codemap_miss_total"] = len(misses)
    out["codemap_budget_refusals_total"] = len(refusals)
    # --- gains (baseline vs codemap, paired by persona + seed)
    out["codemap_gain"] = gains(humans_rows or [])
    return out


def gains(rows):
    def toks(r):
        u = r.get("usage") or {}
        return (u.get("input_tokens", 0) or 0) + (u.get("output_tokens", 0) or 0) + (u.get("cache_read_input_tokens", 0) or 0) \
            + (u.get("cache_creation_input_tokens", 0) or 0)
    base = {(r["persona"], r["seed_id"]): r for r in rows if r.get("mode") == "baseline" and not r.get("is_error")}
    cm = {(r["persona"], r["seed_id"]): r for r in rows if r.get("mode") == "codemap" and not r.get("is_error")}
    pairs = [(base[k], cm[k]) for k in base if k in cm]
    ratios, turns, secs = [], [], []
    for b, c in pairs:
        if toks(c):
            ratios.append(toks(b) / toks(c))
        if c.get("num_turns"):
            turns.append((b.get("num_turns") or 0) - c["num_turns"])
        if c.get("duration_ms"):
            secs.append(((b.get("duration_ms") or 0) - c["duration_ms"]) / 1000.0)
    alone = {}
    minutes = []
    for r in rows:
        rep = r.get("report") or {}
        if r.get("mode") == "codemap" and rep:
            a = rep.get("would_have_found_alone")
            if a:
                alone[a] = alone.get(a, 0) + 1
            if isinstance(rep.get("minutes_saved_estimate"), (int, float)):
                minutes.append(float(rep["minutes_saved_estimate"]))
    return {"n_pairs": len(pairs), "baselines": len(base), "tokens_ratio_mean": _mean(ratios),
            "turns_delta_mean": _mean(turns), "seconds_delta_mean": _mean(secs),
            "would_have_found_alone": alone, "minutes_saved_estimate_mean": _mean(minutes)}


def flatten(doc, date):
    """Scalars as Influx lines (gauges tagged night=<date>) for the quality-over-time dashboard."""
    import time
    ts = time.time_ns()
    lines = []

    def emit(name, value, tags=None):
        """Nightly gauges are codemap_quality_*: they must not collide with the server's running counters."""
        if value is None:
            return
        name = "codemap_quality_" + name[len("codemap_"):] if name.startswith("codemap_") else name
        t = "".join(f",{k}={v}" for k, v in sorted((tags or {}).items()))
        lines.append(f"{name},night={date}{t} value={value} {ts}")
    for k, v in doc.items():
        if k in ("schema", "date", "n") or not k.startswith("codemap_"):
            continue
        if isinstance(v, (int, float)):
            emit(k, v)
        elif isinstance(v, dict) and k not in ("codemap_gain",):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    emit(k, vv, {"key": str(kk).replace(" ", "_")})
    g = doc.get("codemap_gain") or {}
    for kk in ("tokens_ratio_mean", "turns_delta_mean", "seconds_delta_mean", "minutes_saved_estimate_mean", "n_pairs"):
        if isinstance(g.get(kk), (int, float)):
            emit("codemap_gain", g[kk], {"key": kk})
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--humans", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    events = load_jsonl(a.events)
    judge_doc = json.load(open(a.judge, encoding="utf-8")) if a.judge and os.path.exists(a.judge) else None
    humans = load_jsonl(a.humans)
    doc = compute(a.date, events, judge_doc, humans)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(json.dumps({k: doc[k] for k in ("n", "codemap_grounded_rate", "codemap_correct_rate", "codemap_rating_mean",
                                          "codemap_credits_per_correct_answer", "codemap_judge_kappa", "codemap_gain")}, indent=1))
    if a.push:
        from remote import telemetry_push
        lines = flatten(doc, a.date)
        p = telemetry_push.Pusher(None)
        if p.influx_url and p.influx_auth:
            telemetry_push._post(p.influx_url, "\n".join(lines).encode("utf-8"), "text/plain; charset=utf-8", p.influx_auth)
            print(f"pushed {len(lines)} lines")
        else:
            print("push skipped: no Grafana credentials in the environment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
