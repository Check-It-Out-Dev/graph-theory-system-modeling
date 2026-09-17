"""Run the Erdős pairs: every complex problem answered twice, by a general agent with the files only and
by Erdős with the graph and the files.

    PYTHONUTF8=1 python eval/erdos/run_pairs.py --label 2026-09-17 [--only ID] [--arms general,erdos]
                                                [--workspace C:/Users/Norbert/erdos-ws] [--model claude-opus-5]
                                                [--max-turns 100] [--timeout 3600] [--parallel 2] [--dry-run]

The only differences between the arms are the ones under test:

| | general | erdos |
|---|---|---|
| task card (problem + answer contract + phase markers) | same text | same text |
| model, max turns, working directory, file tools (Read, Grep, Glob) | same | same |
| `--restricted` (no user or project instructions or memory, file tools confined to the workspace) | yes | yes |
| MCP servers | none (`--strict-mcp-config`) | the engine over the pack, read-only |
| system prompt addition | none | the erdos-architect skill with the graph map and tool contract attached |

Each run streams to `eval/erdos/runs/<label>/<problem>.<arm>.events.jsonl` (one {"t", "event"} per line,
`t` in seconds since launch) and its phase split lands in `eval/erdos/runs/<label>.json`. A run whose
events file already ends with a result is not repeated. No file in the workspace is changed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
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

TASK_CARD = """You are working in a workspace that holds two checkouts of the checkItOut platform: `backend/` (Spring Boot, Java) and `frontend/` (Angular, TypeScript). Treat the problem the way a senior engineer who must act on the answer would.

PROBLEM
{problem}

HOW TO ANSWER
Phase 1, solve. Write your answer with these sections: Problem; Where it lives today (the modules or subsystems, the files with their paths, the flows that matter); Proposed change (the design and why); Plan (numbered steps, the files each step touches); Risks and invariants (with the tests that guard them); Evidence (each claim the plan rests on, marked FACT if you read it in the code, otherwise INFERENCE or HYPOTHESIS, with its source). Then write the line === ANSWER COMPLETE === on its own.
Phase 2, verify. Check the INFERENCE and HYPOTHESIS claims your plan depends on. Then write the line === VERIFIED === followed by the corrections to your answer, or "no corrections".
Do not modify any file."""


def load_problems(path=PROBLEMS):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def engine_config(pack_dir):
    return {"mcpServers": {"engine": {"command": sys.executable,
                                      "args": [os.path.join(R, "remote", "engine_mcp.py")],
                                      "env": {"CODEMAP_PACK_DIR": os.path.abspath(pack_dir), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}}}}


def command(arm, problem, model, max_turns, files, exe=None):
    """-> argv for one run. `files` holds the paths of the engine MCP config and the Erdős prompt."""
    card = TASK_CARD.format(problem=problem["prompt"])
    cmd = [exe or shutil.which("claude") or "claude", "-p", card, "--model", model, "--restricted",
           "--tools", ",".join(FILE_TOOLS), "--strict-mcp-config", "--permission-mode", "dontAsk",
           "--output-format", "stream-json", "--verbose", "--include-partial-messages",
           "--max-turns", str(max_turns), "--no-session-persistence"]
    if arm == "erdos":
        cmd += ["--mcp-config", files["mcp"], "--allowedTools", ",".join(FILE_TOOLS + ("mcp__engine__*",)),
                "--append-system-prompt-file", files["prompt"]]
    elif arm == "general":
        cmd += ["--allowedTools", ",".join(FILE_TOOLS)]
    else:
        raise ValueError(f"unknown arm {arm}")
    return cmd


def run_one(cmd, cwd, events_path, timeout):
    """Stream one run to its events file; -> seconds. Never raises on a failing run."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["PYTHONUTF8"] = "1"
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
    a = ap.parse_args(argv)
    problems = [p for p in load_problems(a.problems) if not a.only or p["id"] == a.only]
    arms = [x for x in a.arms.split(",") if x]
    run_dir = os.path.join(HERE, "runs", a.label)
    os.makedirs(run_dir, exist_ok=True)
    text = erdos_prompt.assemble()
    files = {"prompt": os.path.join(run_dir, "erdos_prompt.md"), "mcp": os.path.join(run_dir, "engine_mcp.json")}
    with open(files["prompt"], "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(files["mcp"], "w", encoding="utf-8", newline="\n") as f:
        json.dump(engine_config(a.pack), f, indent=1)
    manifest = json.load(open(os.path.join(a.pack, "manifest.json"), encoding="utf-8"))
    meta = {"model": a.model, "max_turns": a.max_turns, "prompt_version": erdos_prompt.version(text),
            "pack_version": manifest.get("pack_version"), "pack_indexed_sha": manifest.get("indexed_sha"),
            "workspace_heads": heads(a.workspace), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
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
