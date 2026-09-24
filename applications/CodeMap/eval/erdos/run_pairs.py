"""Run the Erdős pairs: every complex problem answered twice, by a general agent with the files only and
by Erdős with the raw graph and the files.

    PYTHONUTF8=1 python eval/erdos/run_pairs.py --label <label> [--only ID] [--arms general,erdos]
                                                [--workspace C:/Users/Norbert/erdos-ws] [--model claude-opus-5]
                                                [--max-turns 100] [--timeout 3600] [--parallel 2] [--dry-run]

The only differences between the arms are the ones under test:

| | general | erdos |
|---|---|---|
| task card (problem, answer sections, the completion marker) | same text | same text |
| model, max turns, working directory, file tools (Read, Grep, Glob) | same | same |
| `--restricted` (no user or project instructions, memory, hooks or plugins; file tools confined to the working directories) | yes | yes |
| MCP servers | none (`--strict-mcp-config`) | `graph`: one read-only Cypher tool over the pack (`remote/ladybug_mcp.py`) |
| CLAUDE.md | none | Erdős's XML manual with the LadybugDB introduction, schema and topology, loaded from a context directory passed with `--add-dir` (with `CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD=1`, the one route that loads a CLAUDE.md under `--restricted`, probed 2026-09-17) |

There is no separate verification phase: the card asks for the answer and the marker `=== ANSWER COMPLETE ===`
(the owner, 2026-09-17: check the key files while building the solution, no over-verification). Tokens after
the marker are still counted apart.

Before any Erdős run, a preflight asks a small model, through the same `--restricted --add-dir` route, for the
checksum on the manual's last line, and the runner stops when it does not come back: a CLAUDE.md can stay out of
context silently (a blocking variable, a changed default, a server-side flag). The context directory lives under
the system temp directory, never inside the repository or the workspace, where later sessions would load it. Each run streams to `eval/erdos/runs/<label>/<problem>.<arm>.events.jsonl`
and its phase split lands in `eval/erdos/runs/<label>.json`. No file in the workspace is changed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(R))
sys.path.insert(0, HERE)

import erdos_phases as phases  # noqa: E402
import erdos_prompt  # noqa: E402

PROBLEMS = os.path.join(HERE, "problems.jsonl")
FILE_TOOLS = ("Read", "Grep", "Glob")
KEEP_STREAM = ("message_start", "message_delta")
CLAUDE_MD = erdos_prompt.MANUAL                               # .agents/skills/erdos-architect/CLAUDE.erdos.md
RUN_ENV = {"CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD": "1"}  # both arms; only erdos has an --add-dir to load it from
# Loading switches that must stay unset: each one removes every CLAUDE.md (read from the Claude Code 2.1.273 binary)
BLOCKING_ENV = ("CLAUDE_CODE_DISABLE_CLAUDE_MDS", "CLAUDE_CODE_SAFE_MODE", "CLAUDE_CODE_SIMPLE")
PREFLIGHT_MODEL = "claude-haiku-4-5-20251001"
PREFLIGHT_ASK = ("Your instructions for this session include an XML element named manual_checksum. Reply with the value "
                 "of its value attribute and nothing else, or NONE if there is no such element. Do not use tools.")

TASK_CARD = """You are working in a workspace that holds two checkouts of the checkItOut platform: `backend/` (Spring Boot, Java) and `frontend/` (Angular, TypeScript). Treat the problem the way a senior engineer who must act on the answer would.

PROBLEM
{problem}

HOW TO ANSWER
Write your answer with these sections: Problem; Where it lives today (the modules or subsystems, the files with their paths, the flows that matter); Proposed change (the design and why); Plan (numbered steps, the files each step touches); Risks and invariants (with the tests that guard them); Evidence (each claim the plan rests on, marked FACT if you read it in the code, otherwise INFERENCE or HYPOTHESIS, with its source). End with the line === ANSWER COMPLETE === on its own.
Do not modify any file."""


def load_problems(path=PROBLEMS):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def graph_config(pack_dir):
    return {"mcpServers": {"graph": {"command": sys.executable,
                                     "args": [os.path.join(R, "remote", "ladybug_mcp.py")],
                                     "env": {"CODEMAP_PACK_DIR": os.path.abspath(pack_dir), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}}}}


def context_dir(parent, claude_md_text):
    """A directory holding only Erdős's CLAUDE.md, for --add-dir. Keep `parent` outside the repository and the
    workspace: a CLAUDE.md inside a tree is loaded by any later session that reads files next to it."""
    path = os.path.join(parent, "erdos-context")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "CLAUDE.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(claude_md_text)
    return path


def command(arm, problem, model, max_turns, files, exe=None):
    """-> argv for one run. `files` holds the graph MCP config path and the context directory with Erdős's CLAUDE.md."""
    card = TASK_CARD.format(problem=problem["prompt"])
    cmd = [exe or shutil.which("claude") or "claude", "-p", card, "--model", model, "--restricted",
           "--tools", ",".join(FILE_TOOLS), "--strict-mcp-config", "--permission-mode", "dontAsk",
           "--output-format", "stream-json", "--verbose", "--include-partial-messages",
           "--max-turns", str(max_turns), "--no-session-persistence"]
    if arm == "erdos":
        cmd += ["--mcp-config", files["mcp"], "--allowedTools", ",".join(FILE_TOOLS + ("mcp__graph__*",)),
                "--add-dir", files["context"]]
    elif arm == "general":
        cmd += ["--allowedTools", ",".join(FILE_TOOLS)]
    else:
        raise ValueError(f"unknown arm {arm}")
    return cmd


def child_env():
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY" and k not in BLOCKING_ENV}
    env["PYTHONUTF8"] = "1"
    env.update(RUN_ENV)
    return env


def preflight_command(files, exe=None):
    """-> argv that asks a small model, through Erdős's exact CLAUDE.md channel, for the manual's closing checksum."""
    exe = exe or shutil.which("claude") or "claude"
    return [exe, "-p", PREFLIGHT_ASK, "--model", PREFLIGHT_MODEL, "--restricted", "--tools", "", "--strict-mcp-config",
            "--add-dir", files["context"], "--permission-mode", "dontAsk", "--output-format", "json", "--max-turns", "1",
            "--no-session-persistence"]


def preflight(files, expected, cwd, exe=None, run=subprocess.run, timeout=300):
    """-> (ok, detail). The manual reached the model in full when the model returns the checksum from its last line.
    Guards against every silent way a CLAUDE.md stays out of context (a blocking variable, a changed CLI default,
    a server-side flag that drops project instructions)."""
    try:
        out = run(preflight_command(files, exe), cwd=cwd, env=child_env(), capture_output=True, text=True,
                  encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as ex:
        return False, f"preflight did not run: {ex}"
    try:
        result = json.loads(out.stdout)
    except ValueError:
        return False, f"preflight output unreadable: {out.stdout[:200]!r} {out.stderr[:200]!r}"
    said = (result.get("result") or "").strip()
    usage = result.get("usage") or {}
    context = sum(usage.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    ok = bool(expected) and expected in said and not result.get("is_error")
    return ok, f"expected {expected}, the model said {said[:60]!r}, context {context} tokens"


def reuse_runs(src_dir, dst_dir, problem_ids, arm):
    """Copy finished `<problem>.<arm>.events.jsonl` files from another label; -> the problem ids copied.
    Valid for the general arm across manual versions: its command does not depend on Erdős's manual."""
    copied = []
    for pid in problem_ids:
        src, dst = (os.path.join(d, f"{pid}.{arm}.events.jsonl") for d in (src_dir, dst_dir))
        if finished(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
            copied.append(pid)
    return copied


def run_one(cmd, cwd, events_path, timeout):
    """Stream one run to its events file; -> seconds. Never raises on a failing run."""
    env = child_env()
    t0 = time.monotonic()
    os.makedirs(os.path.dirname(events_path), exist_ok=True)
    with open(events_path, "w", encoding="utf-8", newline="\n") as out:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # NOSONAR - the claude CLI, argv list
                                text=True, encoding="utf-8", errors="replace")
        timer = threading.Timer(timeout, proc.kill)
        timer.start()
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    event = {"type": "stderr", "text": line[:2000]}
                if event.get("type") == "stream_event" and (event.get("event") or {}).get("type") not in KEEP_STREAM:
                    continue  # token chunks: the final usage rides message_delta, the content rides the assistant events
                out.write(json.dumps({"t": round(time.monotonic() - t0, 3), "event": event}, ensure_ascii=False) + "\n")
                out.flush()
            proc.wait()
        finally:
            timer.cancel()
    return round(time.monotonic() - t0, 1)


def finished(events_path):
    if not os.path.exists(events_path):
        return False
    return any(e.get("type") == "result" for _, e in phases.load(events_path))


def heads(workspace):
    out = {}
    for repo in ("backend", "frontend"):
        try:
            out[repo] = subprocess.run(["git", "-C", os.path.join(workspace, repo), "rev-parse", "HEAD"],  # NOSONAR - git, argv list
                                       capture_output=True, text=True, timeout=30).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            out[repo] = None
    return out


def summarize(label, problems, arms, meta):
    rows = []
    for p in problems:
        for arm in arms:
            path = os.path.join(HERE, "runs", label, f"{p['id']}.{arm}.events.jsonl")
            if not os.path.exists(path):
                continue
            s = phases.split(phases.load(path))
            answer_path = os.path.join(HERE, "runs", label, f"{p['id']}.{arm}.answer.md")
            with open(answer_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(s["answer"] + ("\n\n## Corrections after verification\n\n" + s["corrections"] if s["corrections"] else "") + "\n")
            rows.append({"problem": p["id"], "arm": arm, "marker": s["marker"], "verified_marker": s["verified_marker"],
                         "solve": s["solve"], "verify": s["verify"], "total": s["total"], "result": s["result"],
                         "answer_file": os.path.relpath(answer_path, R).replace("\\", "/")})
    doc = {"schema": 1, "label": label, "meta": meta, "rows": rows}
    with open(os.path.join(HERE, "runs", f"{label}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--problems", default=PROBLEMS)
    ap.add_argument("--arms", default="general,erdos")
    ap.add_argument("--workspace", default="C:/Users/Norbert/erdos-ws")
    ap.add_argument("--pack", default=os.path.join(R, "graph", "pack"))
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--context-root", default=os.path.join(tempfile.gettempdir(), "codemap-erdos"))
    ap.add_argument("--reuse-general-from", default=None, help="copy the general arm's finished runs from this label")
    a = ap.parse_args(argv)
    problems = [p for p in load_problems(a.problems) if not a.only or p["id"] == a.only]
    arms = [x for x in a.arms.split(",") if x]
    run_dir = os.path.join(HERE, "runs", a.label)
    os.makedirs(run_dir, exist_ok=True)
    text = open(CLAUDE_MD, encoding="utf-8").read().replace("\r\n", "\n")
    if text != erdos_prompt.assemble():
        print("the committed manual is older than its skill source: run tools/agents/sync_agents.py")
        return 2
    files = {"context": context_dir(os.path.join(a.context_root, a.label), text), "mcp": os.path.join(run_dir, "graph_mcp.json")}
    with open(files["mcp"], "w", encoding="utf-8", newline="\n") as f:
        json.dump(graph_config(a.pack), f, indent=1)
    manifest = json.load(open(os.path.join(a.pack, "manifest.json"), encoding="utf-8"))
    meta = {"model": a.model, "max_turns": a.max_turns, "prompt_version": erdos_prompt.version(text),
            "manual": os.path.relpath(CLAUDE_MD, ROOT).replace(os.sep, "/"), "manual_checksum": erdos_prompt.checksum(text),
            "erdos_channel": "CLAUDE.md via --restricted --add-dir with CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD=1",
            "graph_access": "raw read-only Cypher (remote/ladybug_mcp.py)", "task_card": "single phase, no verification round",
            "pack_version": manifest.get("pack_version"), "pack_indexed_sha": manifest.get("indexed_sha"),
            "workspace_heads": heads(a.workspace), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.reuse_general_from:
        source = os.path.join(HERE, "runs", a.reuse_general_from)
        meta["general_reused_from"] = a.reuse_general_from
        print("general runs reused from", a.reuse_general_from, reuse_runs(source, run_dir, [p["id"] for p in problems], "general"))
    stale = {k: v for k, v in meta["workspace_heads"].items() if v and v != (manifest.get("indexed_sha") or {}).get(k)}
    if stale:
        print("WARNING: workspace heads differ from the pack's indexed commits:", stale)
    jobs = []
    for p in problems:
        for arm in arms:
            events = os.path.join(run_dir, f"{p['id']}.{arm}.events.jsonl")
            if finished(events):
                print("done already:", p["id"], arm)
                continue
            jobs.append((p, arm, command(arm, p, a.model, a.max_turns, files), events))
    if "erdos" in arms and jobs and not a.dry_run:
        ok, detail = preflight(files, meta["manual_checksum"], a.workspace)
        meta["preflight"] = detail
        print("preflight:", "ok" if ok else "FAILED", detail, flush=True)
        if not ok:
            return 3
    if a.dry_run:
        for p, arm, cmd, _ in jobs:
            print(p["id"], arm, " ".join(x if len(x) < 60 else x[:57] + "..." for x in cmd[1:]))
        print(json.dumps(meta, indent=1))
        return 0
    sem = threading.Semaphore(max(1, a.parallel))

    def work(p, arm, cmd, events):
        with sem:
            print(time.strftime("%H:%M:%S"), "start", p["id"], arm, flush=True)
            secs = run_one(cmd, a.workspace, events, a.timeout)
            s = phases.split(phases.load(events))
            print(time.strftime("%H:%M:%S"), "end", p["id"], arm, f"{secs}s", "solve", s["solve"]["tokens_sum"], "verify",
                  s["verify"]["tokens_sum"], "marker", s["marker"], "error", s["result"].get("is_error"), flush=True)

    threads = [threading.Thread(target=work, args=job) for job in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    doc = summarize(a.label, problems, arms, meta)
    print(f"summary: {len(doc['rows'])} runs -> eval/erdos/runs/{a.label}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
