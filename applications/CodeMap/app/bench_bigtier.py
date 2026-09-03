# CodeMap BENCH-2 — the graph-native rung, measured. Same 104-gold ladder and route
# referee as BENCH-1 (bench_promptonly), plus the CONTENT referee doc-06 demanded and
# BENCH-1 never shipped: big models reach true answers by non-canonical routes, so a
# route metric alone undercounts them. Definitions (this file IS the instrument now):
#
#   mentions(text) = corpus entity names found with boundary guards + sub-ids from
#                    'sub-N' / '[N]' surface forms.
#   content F1     = harmonic P/R of mentions(model answer) vs mentions(gold answer),
#                    over EXECUTED golds the model actually answered.
#   number F1      = same over integer literals (count questions live or die here).
#   solid          = every entity the answer mentions appeared in the question or in
#                    a RESULT digest of the session (nothing invented), >= 1 mention.
#
# Rungs:  --rung cypher  the full big_tier prompt (construction story + native cypher)
#         --rung story   ablation: same prompt with the NATIVE QUERY block removed —
#                        isolates "knowing how the graph was built" from "wielding it"
#
# Usage (reuse a warm server):  PYTHONUTF8=1 python app/bench_bigtier.py \
#     --tag qwen80b-native --port 7351 [--rung cypher] [--limit 20]
# Or spawn:  ... --gguf bin/models/<model>.gguf --port 7353

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))
from engine import Engine  # noqa: E402
from datagen import CANONICAL_OVERRIDES, canonical_dsl  # noqa: E402
from loop_runner import result_fp  # noqa: E402
from rung_cpu import SERVER, wait_health  # noqa: E402
import big_tier  # noqa: E402

QFILE = os.path.join(ROOT, "eval", "q", "mfq_all.jsonl")


def build_matcher(names):
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(r"(?<![A-Za-z0-9_./\\-])(" + alt + r")(?![A-Za-z0-9_-])")


def mentions(text, matcher):
    t = text or ""
    out = set(matcher.findall(t))
    out |= {f"sub:{m}" for m in re.findall(r"\bsub[- ]?(\d+)\b", t)}
    out |= {f"sub:{m}" for m in re.findall(r"\[(\d+)\]", t)}
    return out


def nums(text):
    return set(re.findall(r"\b\d+\b", text or ""))


def f1(a, b):
    if not a or not b:
        return 0.0
    i = len(a & b)
    p, r = i / len(a), i / len(b)
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def story_system(engine):
    """The cypher-ablated prompt: construction story + verbs, no native query."""
    s = big_tier.big_system(engine)
    s = s[: s.index("NATIVE QUERY:")] + s[s.index("SURFACE RULES"):]
    s = s.replace("  cypher(<one statement>)  NATIVE read-only query - see below\n", "")
    return s.replace(" or a cypher CONTAINS query", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gguf", default=None, help="spawn a server; omit to reuse --port")
    ap.add_argument("--rung", choices=["cypher", "story"], default="cypher")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(QFILE, encoding="utf-8")]
    if a.limit:
        recs = recs[: a.limit]
    engine = Engine()  # ladybug ON — the native path is the rung
    assert engine.lb is not None, "ladybug offline — the native rung needs the pack DB"
    names = sorted(engine.by_name.keys())
    matcher = build_matcher(names)
    ref_fp = {}
    for r in recs:
        d = CANONICAL_OVERRIDES.get(r["id"]) or canonical_dsl(r, names=names)
        if d:
            ref_fp[r["id"]] = result_fp(engine, d)[1]

    system = big_tier.big_system(engine) if a.rung == "cypher" else story_system(engine)
    l1_evidence = engine.map().get("index") or ""
    srv = None
    if a.gguf:
        srv = subprocess.Popen([SERVER, "-m", a.gguf, "-c", "12288", "--port",
                                str(a.port), "-t", str(a.threads), "--no-webui"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows, t_all = [], time.perf_counter()
    try:
        assert wait_health(a.port, tries=420), "llama-server never became healthy"
        for i, r in enumerate(recs):
            t0 = time.perf_counter()
            traj, terminal, text, results, bt = big_tier.run_big_loop(
                a.port, engine, r["q"], strict=True, system=system)
            dt = time.perf_counter() - t0
            answerable = r.get("gold_status") == "EXECUTED"
            fps = []
            for step in traj:  # route referee: fingerprints of the DSL-verb steps
                if not step.startswith("cypher("):
                    fps.append(result_fp(engine, step)[1])
            hit_fp = bool(answerable and ref_fp.get(r["id"]) in fps)
            ok = (terminal == "answer" and hit_fp) if answerable else (terminal == "pass")
            # grounding evidence = what the model was SHOWN: the L1 index rides
            # in the system prompt (subsystem ids cited from it are not
            # inventions — but the prompt's worked examples do NOT count),
            # then the question and every digest of the session
            evidence = (l1_evidence + " " + r["q"] + " "
                        + " ".join(x["digest"] for x in results))
            am = mentions(text, matcher) if terminal == "answer" else set()
            gm = mentions(r.get("gold_answer"), matcher)
            cy = [x for x in results if x["dsl"].startswith("cypher(")]
            rows.append(dict(
                id=r["id"], status=r.get("gold_status"), terminal=terminal,
                steps=len(traj), backtracks=bt, gold_fp_hit=hit_fp, success=ok,
                cypher_steps=len(cy),
                cypher_errors=sum(1 for x in cy if x["digest"].startswith("ERROR")),
                content_f1=(f1(am, gm) if terminal == "answer" and answerable
                            and r.get("gold_answer") else None),
                number_f1=(f1(nums(text), nums(r.get("gold_answer")))
                           if terminal == "answer" and answerable
                           and r.get("gold_answer") else None),
                solid=(bool(am) and am <= mentions(evidence, matcher)
                       if terminal == "answer" else None),
                seconds=round(dt, 2), traj=traj, answer=(text or "")[:600]))
            if i % 5 == 0:
                print(f"[{i}/{len(recs)}] {r['id']} {terminal} steps={len(traj)} "
                      f"cy={len(cy)} ok={ok} {dt:.1f}s", flush=True)
    finally:
        if srv:
            srv.kill()

    ans = [x for x in rows if x["status"] == "EXECUTED" and ref_fp.get(x["id"])]
    una = [x for x in rows if x["status"] != "EXECUTED"]
    answered = [x for x in rows if x["content_f1"] is not None]
    solids = [x for x in rows if x["solid"] is not None]
    summary = dict(
        rung=a.rung, port=a.port, gguf=a.gguf and os.path.basename(a.gguf),
        n=dict(total=len(rows), answerable=len(ans), unanswerable=len(una),
               answered_with_gold=len(answered)),
        route=dict(
            answerable=round(sum(x["success"] for x in ans) / max(1, len(ans)), 4),
            gold_fp_hit=round(sum(x["gold_fp_hit"] for x in ans) / max(1, len(ans)), 4),
            pass_on_unanswerable=round(
                sum(x["success"] for x in una) / max(1, len(una)), 4)),
        content=dict(
            f1_p50=round(statistics.median(x["content_f1"] for x in answered), 4)
            if answered else None,
            f1_mean=round(statistics.mean(x["content_f1"] for x in answered), 4)
            if answered else None,
            number_f1_p50=round(statistics.median(x["number_f1"] for x in answered), 4)
            if answered else None,
            solid_rate=round(sum(1 for x in solids if x["solid"]) / max(1, len(solids)), 4)),
        cypher=dict(
            sessions_using=round(sum(1 for x in rows if x["cypher_steps"]) / len(rows), 4),
            steps_total=sum(x["cypher_steps"] for x in rows),
            error_rate=round(sum(x["cypher_errors"] for x in rows)
                             / max(1, sum(x["cypher_steps"] for x in rows)), 4)),
        terminals={t: sum(1 for x in rows if x["terminal"] == t)
                   for t in ("answer", "pass", "stall", "invalid")},
        steps=dict(mean=round(statistics.mean(x["steps"] for x in rows), 2),
                   p50=statistics.median(x["steps"] for x in rows)),
        seconds_p50=round(statistics.median(x["seconds"] for x in rows), 2),
        backtrack_rate=round(sum(x["backtracks"] for x in rows) / len(rows), 4),
        wall_minutes=round((time.perf_counter() - t_all) / 60, 1))
    out = os.path.join(ROOT, "training", "data", f"BENCH_{a.tag}.json")
    json.dump(dict(summary=summary, rows=rows), open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))
    print("saved:", out, flush=True)


if __name__ == "__main__":
    main()
