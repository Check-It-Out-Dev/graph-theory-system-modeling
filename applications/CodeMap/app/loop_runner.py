# CodeMap LOOP-level evaluation — the operation the batch rungs cannot see: the trained
# model AUTONOMOUSLY drives multi-step navigation sessions against the live engine
# (find -> verb -> answer / pass), exactly as the product runs. Measures the PIPELINE
# loop-level metrics: task success on gold, steps-to-answer, backtrack rate, stall rate.
#
# Serve contract: ONE user message accumulating "RESULT of <dsl>: <digest_v2>" lines —
# byte-compatible with datagen v2.2 supervision. Grammar-ON llama-server, temperature 0.
#
# Usage: PYTHONUTF8=1 python app/loop_runner.py --gguf bin/models/codemap-lora-r22-q4_k_m.gguf

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))
from dsl import execute, parse, ParseError  # noqa: E402
from engine import Engine  # noqa: E402
from datagen import digest_v2  # noqa: E402  (the serve-contract renderer)
from rung_cpu import GRAMMAR, SERVER, chat, wait_health  # noqa: E402

QFILE = os.path.join(ROOT, "eval", "q", "mfq_all.jsonl")
MAX_STEPS = 6


def result_fp(engine, expr):
    import hashlib
    try:
        r = execute(engine, expr)
        rr = {k: v for k, v in r.items() if k not in ("affordances", "dsl")}
        return r, hashlib.sha256(json.dumps(rr, sort_keys=True, ensure_ascii=False,
                                            default=str).encode()).hexdigest()[:16]
    except Exception as ex:
        return None, f"ERR:{type(ex).__name__}"


def run_loop(port, engine, question):
    """One autonomous session. Returns (trajectory, terminal, answer_text, fps)."""
    user = f"QUESTION: {question}\n(cache: miss)"
    traj, fps, seen = [], [], set()
    backtracks = 0
    for _ in range(MAX_STEPS):
        # 560 not 380: five r3-dpo answers truncated mid-string into parse failures
        gen = chat(port, user, 560).splitlines()[0].strip()
        traj.append(gen)
        try:
            verb, args = parse(gen)
        except ParseError:
            return traj, "invalid", None, fps, backtracks
        if verb == "answer":
            return traj, "answer", args[0], fps, backtracks
        if verb == "pass":
            return traj, "pass", args[0], fps, backtracks
        if gen in seen:
            backtracks += 1
        seen.add(gen)
        res, fp = result_fp(engine, gen)
        fps.append(fp)
        dig = digest_v2(res) if res is not None else f"ERROR: {fp}"
        user += f"\nRESULT of {gen}: {dig}"
    return traj, "stall", None, fps, backtracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--port", type=int, default=7347)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(QFILE, encoding="utf-8")]
    if a.limit:
        recs = recs[: a.limit]
    engine = Engine(use_ladybug=False)
    # engine-vs-engine referee (first run compared engine fps to RECIPE fingerprints —
    # different universes, 0.0 by construction; the instrument lied before the model could)
    sys.path.insert(0, os.path.join(ROOT, "training"))
    from datagen import CANONICAL_OVERRIDES, canonical_dsl
    names = sorted(engine.by_name.keys())
    ref_fp = {}
    for r in recs:
        d = CANONICAL_OVERRIDES.get(r["id"]) or canonical_dsl(r, names=names)
        if d:
            ref_fp[r["id"]] = result_fp(engine, d)[1]

    srv = subprocess.Popen([SERVER, "-m", a.gguf, "-c", "4096", "--port", str(a.port),
                            "-t", str(a.threads), "--no-webui",
                            "--grammar-file", GRAMMAR],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows, t_all = [], time.perf_counter()
    try:
        assert wait_health(a.port), "llama-server never became healthy"
        for i, r in enumerate(recs):
            t0 = time.perf_counter()
            traj, terminal, text, fps, bt = run_loop(a.port, engine, r["q"])
            dt = time.perf_counter() - t0
            answerable = r.get("gold_status") == "EXECUTED"
            hit_fp = bool(answerable and ref_fp.get(r["id"]) in fps)
            ok = (terminal == "answer" and hit_fp) if answerable else (terminal == "pass")
            rows.append(dict(id=r["id"], status=r.get("gold_status"), terminal=terminal,
                             steps=len(traj), backtracks=bt, gold_fp_hit=hit_fp,
                             success=ok, seconds=round(dt, 2), traj=traj,
                             answer=(text or "")[:600], gold=(r.get("gold_answer") or "")[:600]))
            if i % 10 == 0:
                print(f"[{i}/{len(recs)}] {r['id']} {terminal} steps={len(traj)} "
                      f"ok={ok} {dt:.1f}s")
    finally:
        srv.kill()

    ans = [x for x in rows if x["status"] == "EXECUTED" and ref_fp.get(x["id"])]
    una = [x for x in rows if x["status"] != "EXECUTED"]
    summary = dict(
        n=dict(total=len(rows), answerable=len(ans), unanswerable=len(una),
               excluded_no_canonical=sum(1 for x in rows
                                         if x["status"] == "EXECUTED"
                                         and not ref_fp.get(x["id"]))),
        task_success=dict(
            answerable=round(sum(x["success"] for x in ans) / max(1, len(ans)), 4),
            gold_fp_hit=round(sum(x["gold_fp_hit"] for x in ans) / max(1, len(ans)), 4),
            pass_on_unanswerable=round(sum(x["success"] for x in una) / max(1, len(una)), 4)),
        terminals={t: sum(1 for x in rows if x["terminal"] == t)
                   for t in ("answer", "pass", "stall", "invalid")},
        steps=dict(mean=round(statistics.mean(x["steps"] for x in rows), 2),
                   p50=statistics.median(x["steps"] for x in rows),
                   max=max(x["steps"] for x in rows)),
        backtrack_rate=round(sum(x["backtracks"] for x in rows) / len(rows), 4),
        wall_minutes=round((time.perf_counter() - t_all) / 60, 1))
    tag = os.path.splitext(os.path.basename(a.gguf))[0].replace("codemap-", "")
    out = os.path.join(ROOT, "training", "data", f"LOOP_{tag}.json")
    json.dump(dict(summary=summary, rows=rows), open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))
    print("saved:", out)


if __name__ == "__main__":
    main()
