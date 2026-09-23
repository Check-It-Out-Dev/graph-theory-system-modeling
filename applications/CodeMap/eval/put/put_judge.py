"""The judge of the prompt-under-test pipeline, rubric r4: Claude Opus grades one coding run against its task and the
team's WRITTEN conventions, blind to the prompt that produced it.

    ask(run_dir, task, contract, instance, model="opus")  -> (verdict or None, usage, is_error)
    PYTHONUTF8=1 python eval/put/put_judge.py --run <run_dir> --task <id>        grade one run, print the verdict

Blindness (law 3): the judge never sees the candidate prompt, its hash or its iteration. The conventions it grades
against are the canonical rule texts of the seed prompt v1 (what the team wrote), fixed for the whole arc, so a
candidate cannot win by rewording the rules the judge reads. The work summary is built from calls.json (what the
agent queried, read, edited and ran), the test numbers from tests.json; the diff is capped at 40,000 characters,
each file truncated in turn when it is longer.

A verdict carries the rubric id; a change of rubric re-measures the judge's noise and its anchors (law 12).
"""

import argparse
import json
import os
import re
import sys
import tempfile

import put_paths
import put_contract
import put_tasks
import erdos_judge                 # eval/erdos: parse_json_text

RUBRIC = "r4"
RUBRIC_FILE = os.path.join(put_paths.HERE, "judge", "rubric-r4.md")
CRITERIA = ("correctness", "convention_fit", "design_fit", "test_quality", "graph_use")
COUNTS = ("rules_violated", "files_out_of_scope", "parallel_mechanisms")
DIFF_CAP = 40000
RULE = re.compile(r'<rule id="([a-z_]+)">\s*(.*?)\s*</rule>', re.S)


def canonical_rules(instance):
    """-> [(id, text)] from the seed prompt v1: the conventions as the team wrote them, never a candidate's wording.
    The process rules (graph, exemplars, marker) are graded by their own criteria and checks, so they stay out."""
    body = put_contract.prompt_text(instance, "v1")
    skip = {"graph_first", "marker"}
    return [(rid, " ".join(text.split())) for rid, text in RULE.findall(body) if rid not in skip]


def capped_diff(diff, cap=DIFF_CAP):
    if len(diff) <= cap:
        return diff
    parts = re.split(r"(?=^diff --git )", diff, flags=re.M)
    share = max(2000, cap // max(1, len(parts)))
    out = []
    for p in parts:
        out.append(p if len(p) <= share else p[:share] + f"\n[... {len(p) - share} characters of this file omitted ...]\n")
    return "".join(out)[:cap]


def work_summary(calls, limit=60):
    """-> the agent's work in order: graph queries (with their statements), reads, searches, edits, commands."""
    lines = []
    for c in calls[:limit]:
        inp = c.get("input") or {}
        name = c.get("name", "")
        if name.startswith("mcp__graph__"):
            what = "graph query: " + " ".join(str(inp.get("statement", "")).split())[:220]
        elif name in ("Read", "Edit", "Write", "MultiEdit"):
            path = str(inp.get("file_path", "")).replace("\\", "/")
            k = path.find("/src/")
            what = f"{name.lower()}: " + (path[k + 1:] if k >= 0 else path)
        elif name in ("Grep", "Glob"):
            what = f"{name.lower()}: {inp.get('pattern', '')}"
        elif name == "Bash":
            what = "command: " + str(inp.get("command", ""))[:160]
        else:
            what = name
        lines.append(f"{c.get('i')}. {what}" + (" [failed]" if c.get("is_error") else ""))
    if len(calls) > limit:
        lines.append(f"... {len(calls) - limit} more calls")
    return "\n".join(lines)


def test_summary(tests):
    def pair(key):
        v = tests.get(key) or [0, 0]
        return f"{v[0]}/{v[1]}"
    return (f"build: {'green' if tests.get('build_green') else 'red'}; the agent's own unit tests: {pair('own')}; "
            f"hidden acceptance tests: {pair('hidden')}; existing tests of the touched classes: {pair('pass_to_pass')}")


def prompt(task, instance, diff, calls, tests):
    rules = "\n".join(f"- {text}" for _, text in canonical_rules(instance))
    return (f"{put_tasks.card(task)}\n\nTHE TEAM'S WRITTEN CONVENTIONS\n{rules}\n\n"
            f"HOW THE AGENT WORKED\n{work_summary(calls)}\n\nTEST RESULTS (measured by the harness)\n{test_summary(tests)}\n\n"
            f"DIFF\n{capped_diff(diff)}\n\nGrade this change with the rubric. Return only the JSON object.")


def valid(verdict):
    if not isinstance(verdict, dict):
        return False
    for k in CRITERIA:
        v = verdict.get(k)
        if not isinstance(v, (int, float)) or not 1 <= v <= 5:
            return False
    return True


def read_run(run_dir):
    def j(name, default):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            return default
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    with open(os.path.join(run_dir, "diff.patch"), encoding="utf-8") as f:
        diff = f.read()
    return diff, j("calls.json", []), j("tests.json", {})


def ask(run_dir, task, instance, model="opus", runner=None, timeout=900):
    """-> (verdict or None, usage, is_error): one run graded with rubric r4."""
    import claude_cli
    diff, calls, tests = read_run(run_dir)
    with open(RUBRIC_FILE, encoding="utf-8") as f:
        system = f.read()
    fd, sys_path = tempfile.mkstemp(prefix="put-judge-r4-", suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(system)
    try:
        out = claude_cli.run(prompt(task, instance, diff, calls, tests), model=model, role="judge", system_file=sys_path,
                             max_turns=1, timeout=timeout, runner=runner, tools=[])
    finally:
        try:
            os.unlink(sys_path)
        except OSError:
            pass
    verdict = erdos_judge.parse_json_text(out.get("text"))
    ok = valid(verdict)
    if ok:
        verdict["rubric"] = RUBRIC
        verdict["model"] = (out.get("model_usage") and next(iter(out["model_usage"]), None)) or model
    return (verdict if ok else None), out.get("usage"), bool(out.get("is_error")) or not ok


def judge_run(run_dir, task, instance, model="opus", name="verdict-r4.json", force=False):
    """Grade a run once and store the verdict beside it (cached: a stored valid verdict is reused)."""
    path = os.path.join(run_dir, name)
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        if v.get("verdict") is not None:
            return v
    verdict, usage, err = ask(run_dir, task, instance, model)
    rec = {"rubric": RUBRIC, "verdict": verdict, "usage": usage, "is_error": err}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance", default="backend-conventions")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--print-prompt", action="store_true")
    a = ap.parse_args(argv)
    task = next(t for t in put_contract.tasks(a.instance) if t["id"] == a.task)
    if a.print_prompt:
        diff, calls, tests = read_run(a.run)
        print(prompt(task, a.instance, diff, calls, tests))
        return 0
    rec = judge_run(a.run, task, a.instance, a.model, force=a.force)
    print(json.dumps(rec, indent=1))
    return 0 if rec.get("verdict") else 1


if __name__ == "__main__":
    sys.exit(main())
