"""The judge of the prompt-under-test pipeline, rubric r5: Claude Opus grades one coding run against its task and the
team's WRITTEN conventions, blind to the prompt that produced it.

    ask(run_dir, task, contract, instance, model="opus")  -> (verdict or None, usage, is_error)
    PYTHONUTF8=1 python eval/put/put_judge.py --run <run_dir> --task <id>        grade one run, print the verdict

Blindness (law 3): the judge never sees the candidate prompt, its hash or its iteration. The conventions it grades
against are the canonical rule texts of the seed prompt v1 (what the team wrote), fixed for the whole arc, so a
candidate cannot win by rewording the rules the judge reads. The work summary is built from calls.json (what the
agent queried, read, edited and ran), the test numbers from tests.json; the diff is capped at 40,000 characters,
each file truncated in turn when it is longer. BASE CODE holds the unchanged text of the production Java files the
diff modifies (r5): a judge that sees only the diff cannot tell a new method that repeats an existing one from real
reuse, which calibration found on two anchors (judge/reference-scores.json).

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

RUBRIC = "r5"
RUBRIC_FILE = os.path.join(put_paths.HERE, "judge", f"rubric-{RUBRIC}.md")
VERDICT = f"verdict-{RUBRIC}.json"
VERDICT_REPEAT = f"verdict-{RUBRIC}-repeat.json"
CRITERIA = ("correctness", "convention_fit", "design_fit", "test_quality", "graph_use")
COUNTS = ("rules_violated", "files_out_of_scope", "parallel_mechanisms")
DIFF_CAP = 40000
BASE_CAP_FILE = 36000
BASE_CAP = 60000
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


def modified_sources(diff):
    """-> paths of production Java files the diff modifies (not new files, not tests, not resources)."""
    out = []
    for part in re.split(r"(?=^diff --git )", diff, flags=re.M):
        m = re.match(r"diff --git a/(\S+) b/(\S+)", part)
        if not m or "\nnew file mode" in part[:300]:
            continue
        path = m.group(2)
        if path.startswith("src/main/java/") and path.endswith(".java"):
            out.append(path)
    return out


def base_code(diff, base_text):
    """-> the unchanged text of the modified production files, capped per file and in total; "" without a reader."""
    if base_text is None:
        return ""
    blocks, used = [], 0
    for path in modified_sources(diff):
        text = base_text(path)
        if not text:
            continue
        if len(text) > BASE_CAP_FILE:
            text = text[:BASE_CAP_FILE] + f"\n[... {len(text) - BASE_CAP_FILE} characters omitted ...]\n"
        if used + len(text) > BASE_CAP:
            blocks.append(f"--- {path}: omitted (base code budget reached)")
            continue
        blocks.append(f"--- {path} (before the change)\n{text}")
        used += len(text)
    return "\n\n".join(blocks)


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


def prompt(task, instance, diff, calls, tests, base_text=None):
    rules = "\n".join(f"- {text}" for _, text in canonical_rules(instance))
    base = base_code(diff, base_text)
    return (f"{put_tasks.card(task)}\n\nTHE TEAM'S WRITTEN CONVENTIONS\n{rules}\n\n"
            f"HOW THE AGENT WORKED\n{work_summary(calls)}\n\nTEST RESULTS (measured by the harness)\n{test_summary(tests)}\n\n"
            f"DIFF\n{capped_diff(diff)}\n\n" + (f"BASE CODE\n{base}\n\n" if base else "") +
            "Grade this change with the rubric. Return only the JSON object.")


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


def ask(run_dir, task, instance, model="opus", runner=None, timeout=900, base_text=None):
    """-> (verdict or None, usage, is_error): one run graded with the current rubric."""
    import claude_cli
    diff, calls, tests = read_run(run_dir)
    with open(RUBRIC_FILE, encoding="utf-8") as f:
        system = f.read()
    fd, sys_path = tempfile.mkstemp(prefix=f"put-judge-{RUBRIC}-", suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(system)
    try:
        out = claude_cli.run(prompt(task, instance, diff, calls, tests, base_text), model=model, role="judge", system_file=sys_path,
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


def judge_run(run_dir, task, instance, model="opus", name=None, force=False, base_text=None):
    """Grade a run once and store the verdict beside it (cached: a stored valid verdict is reused)."""
    path = os.path.join(run_dir, name or VERDICT)
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        if v.get("verdict") is not None:
            return v
    verdict, usage, err = ask(run_dir, task, instance, model, base_text=base_text)
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
    ap.add_argument("--base-repo", help="a clone of the repository under test: adds BASE CODE (r5)")
    a = ap.parse_args(argv)
    task = next(t for t in put_contract.tasks(a.instance) if t["id"] == a.task)
    base_text = None
    if a.base_repo:
        import put_diff
        base_text = put_diff.git_base_reader(a.base_repo, put_contract.load(a.instance)["base_sha"])
    if a.print_prompt:
        diff, calls, tests = read_run(a.run)
        print(prompt(task, a.instance, diff, calls, tests, base_text))
        return 0
    rec = judge_run(a.run, task, a.instance, a.model, force=a.force, base_text=base_text)
    print(json.dumps(rec, indent=1))
    return 0 if rec.get("verdict") else 1


if __name__ == "__main__":
    sys.exit(main())
