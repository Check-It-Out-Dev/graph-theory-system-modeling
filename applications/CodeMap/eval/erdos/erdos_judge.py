"""Judge the Erdős pairs: deterministic checks against the answer key, then one blind comparison per problem.

    PYTHONUTF8=1 python eval/erdos/erdos_judge.py --label 2026-09-17 [--workspace C:/Users/Norbert/erdos-ws] [--model claude-opus-5]

Deterministic, per answer:
- `must_find` recall: the key's files named in the answer (by file name, case-insensitive).
- unknown files: file names the answer mentions that exist nowhere in the workspace (a hallucination
  signal that needs no model).

Blind, per problem: the answer key (key facts, gaps a strong answer discovers, invariants, acceptable
designs, red flags, and since rubric r2 the architecture the project already uses for the concern) and the
two answers as A and B, with the order fixed by a hash of the problem id so
reruns stay comparable. The judge never learns which answer used the graph. It scores each answer and
says whether the two are equivalent in substance and which is better. Judge tokens are recorded apart
from the arms'.

Rubric r2 (2026-09-17, the owner: judge how well a design sticks to the architecture already in the project)
adds `architecture_fit` and `patterns_followed`, and writes `runs/<label>.judge-r2.json`; rubric r1 files
(`runs/<label>.judge.json`) stay as they were recorded. Rubric r2 also neutralises, in both answers alike, the
wording that tells which arm used the graph: bracketed subsystem ids such as `[11]` go, and "graph" becomes
"dependency analysis". Answers written with the Erdős 2.0 manual and earlier carry that wording; from 2.1 on the
manual keeps it out. The deterministic checks read the original answers; the judge file counts the replacements.
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

_ID = r"\[\d{1,3}\]"
_IDS = _ID + r"(?:\s*(?:/|,|and)\s*" + _ID + r")*"
ARM_TELLS = [
    (re.compile(r"\bsubsystems?\s+" + _IDS, re.I), ""),                       # "subsystem [11]", "subsystems [170], [174]"
    (re.compile(r"(?<![\w`\]])" + _IDS + r"[ \t]?"), ""),                       # "[5] ", "[173]/[178]"; never code such as a[0]
    (re.compile(r"\bThe graph\b"), "The dependency analysis"),
    (re.compile(r"\bthe graph\b", re.I), "the dependency analysis"),
    (re.compile(r"\bgraph\b", re.I), "dependency analysis"),
]
TIDY = [
    (re.compile(r"\(\s*[,/]?\s*\)"), ""),                                      # parentheses the removals emptied
    (re.compile(r"\(\s*,\s*"), "("),
    (re.compile(r"(?<=\S)[ \t]+(?=[,.:;)])"), ""),                             # a space left before punctuation
]
FILE_RX = re.compile(r"[A-Za-z0-9_.\-]+\.(?:java|ts|html|scss|css|yml|yaml|xml|properties|feature|sql|js|mjs|json)\b")
SKIP_DIRS = {".git", "node_modules", "target", "dist", "build", ".angular"}
RUBRIC = "r2"

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
- patterns_followed: how many of the key's architecture entries the design reuses or extends correctly.
- architecture_fit 1-5: 5 the design builds on the mechanisms the project already uses for this concern (the key's architecture entries: switches and profiles, scheduled jobs, events, adapters, persistence, authorization, frontend clients) and adds no parallel mechanism; 3 mostly fits but duplicates one existing mechanism or bypasses one that applies; 1 ignores or duplicates the existing architecture.
- overall 1-5: would you act on this answer; weigh correctness, design and architecture_fit most.

Then: equivalent (true when both reach substantially the same conclusions and plan), better ("A", "B" or "tie"), why (at most 80 words).
Reply with ONE JSON object only, no prose, no code fence:
{"A": {...scores...}, "B": {...scores...}, "equivalent": true, "better": "tie", "why": "..."}"""


def neutralize(text):
    """-> (text, replacements): the answer without the wording that reveals the graph arm."""
    count = 0
    for rx, repl in ARM_TELLS:
        text, n = rx.subn(repl, text)
        count += n
    if count:
        for rx, repl in TIDY:
            text = rx.sub(repl, text)
    return text, count


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
    import take_gold
    g = problem.get("gold") or take_gold.load(problem["id"]) or {}
    return {k: g.get(k) for k in ("must_find", "key_facts", "gaps", "invariants", "good_designs", "red_flags", "architecture")}


def judge_path(label, rubric=RUBRIC):
    return os.path.join(HERE, "runs", f"{label}.judge.json" if rubric == "r1" else f"{label}.judge-{rubric}.json")


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
    out = {"schema": 1, "label": a.label, "model": a.model, "rubric": RUBRIC, "problems": []}
    for p in run_pairs.load_problems():
        if a.only and p["id"] != a.only:
            continue
        if not all((p["id"], arm) in by for arm in ("general", "erdos")):
            continue
        answers = {arm: open(os.path.join(R, by[(p["id"], arm)]["answer_file"]), encoding="utf-8").read() for arm in ("general", "erdos")}
        must = key_for_judge(p).get("must_find") or []
        det = {arm: {"must_find": recall(answers[arm], must), "files": unknown_files(answers[arm], index)} for arm in answers}
        order = blind_order(p["id"])
        shown = {arm: neutralize(answers[arm]) for arm in answers}
        verdict, usage, err = ask(p, shown[order[0]][0], shown[order[1]][0], a.model)
        mapped = None
        if verdict:
            mapped = {"scores": {order[0]: verdict.get("A"), order[1]: verdict.get("B")}, "equivalent": verdict.get("equivalent"),
                      "better": {"A": order[0], "B": order[1]}.get(verdict.get("better"), "tie"), "why": verdict.get("why")}
        out["problems"].append({"problem": p["id"], "blind_order": {"A": order[0], "B": order[1]}, "deterministic": det,
                                "neutralized": {arm: shown[arm][1] for arm in shown},
                                "judge": mapped, "judge_usage": usage, "judge_error": err})
        print(p["id"], "better:", (mapped or {}).get("better"), "equivalent:", (mapped or {}).get("equivalent"),
              "recall g/e:", det["general"]["must_find"]["recall"], det["erdos"]["must_find"]["recall"], flush=True)
    with open(judge_path(a.label), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
