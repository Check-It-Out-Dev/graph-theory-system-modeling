"""Judge the Erdős pairs: deterministic checks against the answer key, then one blind comparison per problem.

    PYTHONUTF8=1 python eval/erdos/erdos_judge.py --label 2026-09-17 [--workspace C:/Users/Norbert/erdos-ws] [--model claude-opus-5]

Deterministic, per answer:
- `must_find` recall: the key's files named in the answer (by file name, case-insensitive).
- unknown files: file names the answer mentions that exist nowhere in the workspace (a hallucination
  signal that needs no model).

Blind, per problem: the answer key (key facts, gaps a strong answer discovers, invariants, acceptable
designs, red flags) and the two answers as A and B, with the order fixed by a hash of the problem id so
reruns stay comparable. The judge never learns which answer used the graph. It scores each answer and
says whether the two are equivalent in substance and which is better. Judge tokens are recorded apart
from the arms'.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "app"))
sys.path.insert(0, HERE)

FILE_RX = re.compile(r"[A-Za-z0-9_.\-]+\.(?:java|ts|html|scss|css|yml|yaml|xml|properties|feature|sql|js|mjs|json)\b")
SKIP_DIRS = {".git", "node_modules", "target", "dist", "build", ".angular"}

SYSTEM = """You are a principal engineer grading two answers to the same hard architectural problem about one codebase.
You have an answer key built from the source code. Grade only against the key and the answers; do not reward length or confidence.

Scores per answer (integers):
- must_find_hits: how many of the key's must_find files the answer identifies (by name or unmistakable description).
- key_facts_supported: how many key facts the answer states correctly or relies on correctly.
- gaps_found: how many of the key's gaps the answer discovers.
- red_flags_made: how many of the key's red flags (wrong claims) the answer makes, plus other claims the key contradicts.
- correctness 1-5: 5 no false claims about the code; 3 minor errors that do not change the plan; 1 errors that would mislead the implementation.
- design 1-5: 5 respects every invariant and addresses the gaps with a sound design; 3 workable but misses an invariant or a gap; 1 unsound.
- plan 1-5: 5 ordered, complete, names the files per step and the tests; 3 partly actionable; 1 not actionable.
- overall 1-5: would you act on this answer.

Then: equivalent (true when both reach substantially the same conclusions and plan), better ("A", "B" or "tie"), why (at most 80 words).
Reply with ONE JSON object only, no prose, no code fence:
{"A": {...scores...}, "B": {...scores...}, "equivalent": true, "better": "tie", "why": "..."}"""


def workspace_index(workspace):
    names = set()
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        names.update(f.lower() for f in files)
    return names


def recall(answer, must_find):
    text = answer.lower()
    hits = [m["name"] for m in must_find if m["name"].lower() in text]
    return {"hits": hits, "n": len(must_find), "recall": round(len(hits) / len(must_find), 3) if must_find else None}


def unknown_files(answer, index):
    mentioned = sorted(set(m.lower() for m in FILE_RX.findall(answer)))
    unknown = [m for m in mentioned if m not in index]
    return {"mentioned": len(mentioned), "unknown": unknown}


def blind_order(problem_id):
    """-> ("general", "erdos") or ("erdos", "general") as (A, B), fixed per problem id."""
    return ("general", "erdos") if int(hashlib.sha256(problem_id.encode()).hexdigest(), 16) % 2 == 0 else ("erdos", "general")


def key_for_judge(problem):
    g = problem.get("gold") or {}
    return {k: g.get(k) for k in ("must_find", "key_facts", "gaps", "invariants", "good_designs", "red_flags")}


def judge_prompt(problem, answer_a, answer_b):
    return ("PROBLEM\n" + problem["prompt"] + "\n\nANSWER KEY (from the source code)\n" +
            json.dumps(key_for_judge(problem), ensure_ascii=False, indent=1) +
            "\n\nANSWER A\n" + answer_a + "\n\nANSWER B\n" + answer_b)


def parse_json_text(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def ask(problem, answer_a, answer_b, model, runner=None, timeout=900):
    import claude_cli
    fd, sys_path = tempfile.mkstemp(prefix="erdos-judge-", suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(SYSTEM)
    try:
        out = claude_cli.run(judge_prompt(problem, answer_a, answer_b), model=model, role="judge", system_file=sys_path,
                             max_turns=1, timeout=timeout, runner=runner, tools=[])
    finally:
        try:
            os.unlink(sys_path)
        except OSError:
            pass
    return parse_json_text(out.get("text")), out.get("usage"), out.get("is_error")


def main(argv=None):
    import run_pairs
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--workspace", default="C:/Users/Norbert/erdos-ws")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--only", default=None)
    a = ap.parse_args(argv)
    runs = json.load(open(os.path.join(HERE, "runs", f"{a.label}.json"), encoding="utf-8"))
    by = {(r["problem"], r["arm"]): r for r in runs["rows"]}
    index = workspace_index(a.workspace)
    out = {"schema": 1, "label": a.label, "model": a.model, "problems": []}
    for p in run_pairs.load_problems():
        if a.only and p["id"] != a.only:
            continue
        if not all((p["id"], arm) in by for arm in ("general", "erdos")):
            continue
        answers = {arm: open(os.path.join(R, by[(p["id"], arm)]["answer_file"]), encoding="utf-8").read() for arm in ("general", "erdos")}
        must = (p.get("gold") or {}).get("must_find") or []
        det = {arm: {"must_find": recall(answers[arm], must), "files": unknown_files(answers[arm], index)} for arm in answers}
        order = blind_order(p["id"])
        verdict, usage, err = ask(p, answers[order[0]], answers[order[1]], a.model)
        mapped = None
        if verdict:
            mapped = {"scores": {order[0]: verdict.get("A"), order[1]: verdict.get("B")}, "equivalent": verdict.get("equivalent"),
                      "better": {"A": order[0], "B": order[1]}.get(verdict.get("better"), "tie"), "why": verdict.get("why")}
        out["problems"].append({"problem": p["id"], "blind_order": {"A": order[0], "B": order[1]}, "deterministic": det,
                                "judge": mapped, "judge_usage": usage, "judge_error": err})
        print(p["id"], "better:", (mapped or {}).get("better"), "equivalent:", (mapped or {}).get("equivalent"),
              "recall g/e:", det["general"]["must_find"]["recall"], det["erdos"]["must_find"]["recall"], flush=True)
    with open(os.path.join(HERE, "runs", f"{a.label}.judge.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
