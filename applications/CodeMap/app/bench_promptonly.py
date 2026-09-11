# CodeMap BENCH-1 — can a BIG raw instruct model learn CMDSL from the prompt alone?
# Same 104-gold ladder, same engine-vs-engine referee, same success rule as
# loop_runner.py — only the model is not ours and the prompt does the teaching.
#
# Grid axes:  --grammar on|off   (GBNF forces the DSL surface vs. liberal decoding)
# The teaching prompt = frozen MASTER (the serve contract) + a guard preamble for
# models that have never seen it. Grammar-off outputs pass through clean_line()
# (strip code fences / bullets / "CMDSL:" labels) before the strict parser —
# tolerance in the harness, never in the referee.
#
# Usage:
#   PYTHONUTF8=1 python app/bench_promptonly.py --gguf bin/models/<model>.gguf \
#       --grammar off --tag qwen30b-a3b-nogram [--limit 20]

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))
from dsl import parse, ParseError  # noqa: E402
from engine import Engine  # noqa: E402
from datagen import digest_v2, CANONICAL_OVERRIDES, canonical_dsl  # noqa: E402
from rung_cpu import GRAMMAR, SERVER, MASTER, wait_health  # noqa: E402
from loop_runner import result_fp, MAX_STEPS  # noqa: E402

QFILE = os.path.join(ROOT, "eval", "q", "mfq_all.jsonl")

GUARD = (
    "You are a code-graph navigator. You are NOT fine-tuned for this protocol, "
    "so follow it to the letter. Respond with EXACTLY ONE line of CMDSL per "
    "turn - no prose, no markdown, no explanations, no code fences. If results "
    "in context already support the answer, emit answer(\"...\"); if the graph "
    "cannot support one, emit pass(\"reason\"). Never invent names: only use "
    "identifiers visible in the question or in RESULT lines.\n\n"
    "SURFACE RULES a fine-tuned navigator knows and you must copy exactly:\n"
    "- enter takes a BARE integer: enter(11) - NEVER enter(sub 11), enter([11]) "
    "or enter(\"11\"). Result lines print ids like sub-11; strip the prefix.\n"
    "- find takes ONE atomic term (a single identifier or word copied from the "
    "question): find(Fakturownia), find(cron) - NEVER a multi-word phrase. If "
    "one term returns nothing, try a DIFFERENT single term before passing.\n"
    "- answer(...) and pass(...) must be ONE physical line - no newlines inside "
    "the quotes; join list items with '; '.\n"
    "- A result line starting with ERROR means that exact expression failed - "
    "do not repeat it; change the verb or the argument.\n"
    "- Never repeat an expression you already ran; its RESULT is already in "
    "context.\n"
    "- Arguments are BARE names: impact(Foo.java) - never copy parenthetical "
    "annotations from results into arguments.\n"
    "- If two find() terms return nothing, try a camelCase or partial "
    "identifier (find(StepUp)) before passing.\n\n"
    "CYPHER ANALOGY - you already know graph query languages; CMDSL verbs are "
    "shorthands for Cypher over a code graph (nodes: files with name/layer/sub; "
    "edges: DEP):\n"
    "  find(T)    ~ MATCH (n) WHERE n.name CONTAINS 'T' RETURN n\n"
    "  impact(F)  ~ MATCH (d)-[:DEP]->(f {name:'F'}) RETURN d   (who depends on F)\n"
    "  flow(F)    ~ MATCH (f {name:'F'})-[:DEP]->(u) RETURN u   (what F uses)\n"
    "  enter(11)  ~ MATCH (n {sub:11}) RETURN n                 (open subsystem 11)\n"
    "  seam(3,11) ~ MATCH (a {sub:3})-[:DEP]-(b {sub:11}) RETURN a,b\n"
    "  cohort(F)  ~ files that historically CHANGE TOGETHER with F\n"
    "  spine(11)  ~ the load-bearing dependency chain inside sub 11\n"
    "  map()      ~ the L1 index of all subsystems\n"
    "Pick the verb whose Cypher shape matches the question; chain verbs like "
    "MATCH clauses, reading each RESULT before the next.\n\n"
    "HARD BUDGET: you get at most 6 expressions total; each user turn is "
    "prefixed STEP k/6. Do not spend steps reading extra files for confidence "
    "- by STEP 5 you MUST commit: answer(...) grounded in the RESULTs you "
    "already have, or pass(...) if the graph truly cannot answer. An answer "
    "from partial evidence beats a stall.\n\n"
    "EXAMPLE SESSION (format only - your entities will differ):\n"
    "QUESTION: what depends on FooService.java?\n"
    "-> impact(FooService.java)\n"
    "RESULT of impact(FooService.java): 12 dependents ... BarController.java ...\n"
    "-> answer(\"12 dependency edges point at FooService.java; heaviest: "
    "BarController.java ...\")\n\n"
)


def clean_line(gen):
    """Liberal-decoding tolerance: fences, labels, bullets -> the one DSL line.
    Multi-line answer(...)/pass(...) bodies are collapsed to one line — big
    instructs love newline lists; the referee never sees the difference."""
    txt = gen.strip()
    txt = re.sub(r"(?:^```[a-z]*\n?)|(?:```$)", "", txt, flags=re.M).strip()
    lines = txt.splitlines()
    for i, line in enumerate(lines):
        line = line.strip().lstrip("-*> ").strip()
        line = re.sub(r"^(CMDSL|DSL|Action|Step \d+)\s*[:>]\s*", "", line, flags=re.I)
        if re.match(r"^(answer|pass)\(", line) and line.count('"') % 2 == 1:
            rest = " ".join(x.strip() for x in lines[i + 1:])
            line = (line + " " + rest).strip()
        if re.match(r"^[a-z_]+\(", line):
            # Cypher-primed models close statements with ';' — strip it (b10155
            # forensics: 17/104 correct answers rejected for one character).
            return re.sub(r"\)\s*;\s*$", ")", line)
    return lines[0].strip() if lines else ""


def chat(port, system, user, max_tokens):
    body = json.dumps({"model": "bench", "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read().decode("utf-8"))
    return out["choices"][0]["message"]["content"].strip()


def run_loop_liberal(port, engine, question, system, grammar_on):
    user = f"QUESTION: {question}\n(cache: miss)"
    traj, fps, seen, backtracks = [], [], set(), 0
    for step in range(MAX_STEPS):
        user_now = user + f"\nSTEP {step + 1}/{MAX_STEPS}:"
        raw = chat(port, system, user_now, 560)
        gen = raw.splitlines()[0].strip() if grammar_on else clean_line(raw)
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
            user += (f"\nNOTE: you already ran {gen}; its RESULT is above. "
                     "Emit a DIFFERENT expression, or answer()/pass() now.")
            continue
        seen.add(gen)
        res, fp = result_fp(engine, gen)
        fps.append(fp)
        dig = digest_v2(res) if res is not None else f"ERROR: {fp}"
        user += f"\nRESULT of {gen}: {dig}"
    return traj, "stall", None, fps, backtracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--grammar", choices=["on", "off"], default="off")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=7349)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(QFILE, encoding="utf-8")]
    if a.limit:
        recs = recs[: a.limit]
    engine = Engine(use_ladybug=False)
    names = sorted(engine.by_name.keys())
    ref_fp = {}
    for r in recs:
        d = CANONICAL_OVERRIDES.get(r["id"]) or canonical_dsl(r, names=names)
        if d:
            ref_fp[r["id"]] = result_fp(engine, d)[1]

    system = GUARD + MASTER
    cmd = [SERVER, "-m", a.gguf, "-c", "8192", "--port", str(a.port),
           "-t", str(a.threads), "--no-webui"]
    if a.grammar == "on":
        cmd += ["--grammar-file", GRAMMAR]
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows, t_all = [], time.perf_counter()
    try:
        assert wait_health(a.port, tries=300), "llama-server never became healthy"
        for i, r in enumerate(recs):
            t0 = time.perf_counter()
            traj, terminal, text, fps, bt = run_loop_liberal(
                a.port, engine, r["q"], system, a.grammar == "on")
            dt = time.perf_counter() - t0
            answerable = r.get("gold_status") == "EXECUTED"
            hit_fp = bool(answerable and ref_fp.get(r["id"]) in fps)
            ok = (terminal == "answer" and hit_fp) if answerable else (terminal == "pass")
            rows.append(dict(id=r["id"], status=r.get("gold_status"), terminal=terminal,
                             steps=len(traj), backtracks=bt, gold_fp_hit=hit_fp,
                             success=ok, seconds=round(dt, 2), traj=traj,
                             answer=(text or "")[:600]))
            if i % 5 == 0:
                print(f"[{i}/{len(recs)}] {r['id']} {terminal} steps={len(traj)} "
                      f"ok={ok} {dt:.1f}s", flush=True)
    finally:
        srv.kill()

    ans = [x for x in rows if x["status"] == "EXECUTED" and ref_fp.get(x["id"])]
    una = [x for x in rows if x["status"] != "EXECUTED"]
    summary = dict(
        model=os.path.basename(a.gguf), grammar=a.grammar,
        n=dict(total=len(rows), answerable=len(ans), unanswerable=len(una)),
        task_success=dict(
            answerable=round(sum(x["success"] for x in ans) / max(1, len(ans)), 4),
            gold_fp_hit=round(sum(x["gold_fp_hit"] for x in ans) / max(1, len(ans)), 4),
            pass_on_unanswerable=round(
                sum(x["success"] for x in una) / max(1, len(una)), 4)),
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
