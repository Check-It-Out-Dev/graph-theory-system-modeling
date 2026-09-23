"""One coding run: a Claude Sonnet agent solves one task in a fresh worktree of the repository under test, with the
candidate prompt as its CLAUDE.md and the code graph as its one MCP tool; the runner then captures what it did.

    execute(task, body, run_dir, source, contract, instance) -> run record (also written as meta.json)

Per run, under `runs/<label>/<candidate sha16>/<task>.r<k>/`:
    events.jsonl      the stream-json of the session, one {"t", "event"} per line (not committed: sha256 in meta)
    trace.md          the tool calls in order with short results (what the checks and the judge read; committed)
    answer.md         the agent's final message
    diff.patch        every change against the base commit, new files included
    status.txt        `git status --porcelain`
    tests.json        test-compile, the agent's own unit tests, hidden + pass-to-pass acceptance tests
    build.log, test.log, surefire/  Maven output (surefire XML not committed)
    meta.json         model id, prompt version, seconds, usage, output-token budget, worktree head, file hashes

The session is `claude -p` on the subscription (the child never sees ANTHROPIC_API_KEY), `--restricted` (no user
or project settings, file tools confined to the worktree), the tools of the contract and a narrow Bash allowlist
(S0: Bash is not confined by --restricted; the allowlist under dontAsk denies mutating commands outside it). The
output-token budget is the per-run limit: the runner sums output tokens from the stream and ends a session that
passes it, recording `budget_exhausted`.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import put_paths
import put_maven
import put_prompt
import put_tasks
import put_worktree
import run_pairs            # eval/erdos: graph_config, preflight, child_env
import erdos_phases         # eval/erdos: load, split, SOLVED

KEEP_STREAM = ("message_start", "message_delta")
FILE_EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")


def context_dir(parent, text):
    """A directory holding only the candidate's CLAUDE.md, for --add-dir; outside the repository and the worktree."""
    path = os.path.join(parent, "put-context")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "CLAUDE.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def command(card, files, runner_cfg, model, exe=None):
    """-> argv of one coding session."""
    return [exe or shutil.which("claude") or "claude", "-p", card, "--model", model, "--restricted",
            "--tools", ",".join(runner_cfg["tools"]),
            "--allowedTools", ",".join(runner_cfg["allowed_tools"]),
            "--mcp-config", files["mcp"], "--strict-mcp-config",
            "--add-dir", files["context"],
            "--permission-mode", "dontAsk", "--output-format", "stream-json", "--verbose",
            "--include-partial-messages", "--max-turns", str(runner_cfg["max_turns"]), "--no-session-persistence"]


def child_env(role="coder", prompt_version=None, label=None):
    env = run_pairs.child_env()
    for k in ("ANTHROPIC_AUTH_TOKEN", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
        env.pop(k, None)
    base = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "service.name=codemap-put")      # set by put_telemetry.enable
    attrs = [base, f"role={role}"]
    if prompt_version:
        attrs.append(f"prompt_version={prompt_version}")
    if label and f"campaign={label}" not in base:
        attrs.append(f"campaign={label}")
    env["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(attrs)
    return env


class Budget:
    """Output tokens across the session's messages: the last output_tokens of each message, summed."""

    def __init__(self, limit):
        self.limit = limit
        self.per_message = {}
        self.current = None

    def see(self, ev):
        if ev.get("type") != "stream_event":
            return False
        e = ev.get("event") or {}
        if e.get("type") == "message_start":
            self.current = (e.get("message") or {}).get("id") or f"m{len(self.per_message)}"
            usage = (e.get("message") or {}).get("usage") or {}
            self.per_message[self.current] = usage.get("output_tokens", 0) or 0
        elif e.get("type") == "message_delta" and self.current is not None:
            usage = e.get("usage") or {}
            if "output_tokens" in usage:
                self.per_message[self.current] = usage["output_tokens"] or 0
        return self.limit is not None and self.total() > self.limit

    def total(self):
        return sum(self.per_message.values())


def run_session(cmd, cwd, events_path, timeout, output_budget, env, popen=subprocess.Popen):
    """Stream one session into events_path. -> {"seconds", "exit", "budget_exhausted", "timed_out", "output_tokens"}."""
    t0 = time.time()
    budget = Budget(output_budget)
    state = {"budget": False, "timeout": False}
    proc = popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                 text=True, encoding="utf-8", errors="replace", bufsize=1)

    def watchdog():
        if proc.poll() is None:
            state["timeout"] = True
            proc.kill()

    timer = threading.Timer(timeout, watchdog)
    timer.start()
    try:
        with open(events_path, "w", encoding="utf-8", newline="\n") as out:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                over = budget.see(ev)
                keep = ev.get("type") != "stream_event" or (ev.get("event") or {}).get("type") in KEEP_STREAM
                if keep:
                    out.write(json.dumps({"t": round(time.time() - t0, 2), "event": ev}, ensure_ascii=False) + "\n")
                if over and not state["budget"]:
                    state["budget"] = True
                    out.write(json.dumps({"t": round(time.time() - t0, 2),
                                          "event": {"type": "budget_exhausted", "output_tokens": budget.total(),
                                                    "limit": output_budget}}) + "\n")
                    proc.kill()
                    break
        proc.wait(timeout=60)
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
    err = ""
    try:
        err = proc.stderr.read() if proc.stderr else ""
    except (OSError, ValueError):
        pass
    return {"seconds": round(time.time() - t0, 1), "exit": proc.returncode, "budget_exhausted": state["budget"],
            "timed_out": state["timeout"], "output_tokens": budget.total(), "stderr_tail": (err or "")[-800:]}


def usage_of(events):
    """-> usage totals and the resolved model from a loaded event list [(t, event)]."""
    model, result = None, {}
    for _, ev in events:
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            model = ev.get("model")
        if ev.get("type") == "result":
            result = ev
    u = result.get("usage") or {}
    return {"model": model, "num_turns": result.get("num_turns"), "is_error": result.get("is_error"),
            "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
            "cache_read_input_tokens": u.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
            "total_cost_usd": result.get("total_cost_usd"), "duration_ms": result.get("duration_ms")}


def trace_md(events, max_result=300):
    """-> the tool calls in order, one line each with a short result: the record checks and the judge read."""
    calls, results = [], {}
    for _, ev in events:
        msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
        content = msg.get("content") if isinstance(msg.get("content"), list) else []
        if ev.get("type") == "assistant":
            for c in content:
                if c.get("type") == "tool_use":
                    calls.append(c)
        elif ev.get("type") == "user":
            for c in content:
                if c.get("type") == "tool_result":
                    txt = c.get("content")
                    if isinstance(txt, list):
                        txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
                    results[c.get("tool_use_id")] = ((txt or "").replace("\n", " ")[:max_result], bool(c.get("is_error")))
    lines = []
    for i, c in enumerate(calls, 1):
        inp = c.get("input") or {}
        arg = inp.get("statement") or inp.get("command") or inp.get("file_path") or inp.get("pattern") or ""
        res, err = results.get(c.get("id"), ("", False))
        lines.append(f"{i}. {c.get('name')} {str(arg).replace(chr(10), ' ')[:400]}" + (" [error]" if err else "")
                     + (f" -> {res}" if res else ""))
    return "\n".join(lines) + "\n"


def calls_of(events, marker=erdos_phases.SOLVED):
    """-> the tool calls in order, compact enough to commit: name, input (file contents reduced to their size),
    whether it failed, the head of its result, and whether it came after the completion marker."""
    calls, results, marker_seen = [], {}, False
    for _, ev in events:
        msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
        content = msg.get("content") if isinstance(msg.get("content"), list) else []
        if ev.get("type") == "assistant":
            for c in content:
                if c.get("type") == "text" and marker in (c.get("text") or ""):
                    marker_seen = True
                if c.get("type") == "tool_use":
                    inp = dict(c.get("input") or {})
                    for k in ("content", "new_string", "old_string"):
                        if isinstance(inp.get(k), str):
                            inp[k] = f"<{len(inp[k])} chars>"
                    if isinstance(inp.get("edits"), list):
                        inp["edits"] = f"<{len(inp['edits'])} edits>"
                    calls.append({"i": len(calls) + 1, "id": c.get("id"), "name": c.get("name"), "input": inp,
                                  "after_marker": marker_seen})
        elif ev.get("type") == "user":
            for c in content:
                if c.get("type") == "tool_result":
                    txt = c.get("content")
                    if isinstance(txt, list):
                        txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
                    results[c.get("tool_use_id")] = ((txt or "")[:200], bool(c.get("is_error")))
    for c in calls:
        head, err = results.get(c.pop("id"), ("", False))
        c["is_error"] = err
        c["result_head"] = head
    return calls


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir_for(runs_root, label, body, task_id, rep):
    return os.path.join(runs_root, label, put_prompt.sha16(body), f"{task_id}.r{rep}")


def finished(run_dir):
    return os.path.exists(os.path.join(run_dir, "meta.json"))


def execute(task, body, run_dir, source, contract, instance, label=None, workspace_root=None, exe=None,
            popen=subprocess.Popen, keep_worktree=False):
    """Run one (candidate, task, replicate) end to end and write its artifacts. -> the meta record."""
    os.makedirs(run_dir, exist_ok=True)
    cfg = contract["runner"]
    rendered = put_prompt.render(body, instance)
    version = put_prompt.version(body)
    tmp = tempfile.mkdtemp(prefix="put-ctx-")
    files = {"context": context_dir(tmp, rendered), "mcp": os.path.join(tmp, "graph_mcp.json")}
    with open(files["mcp"], "w", encoding="utf-8") as f:
        json.dump(run_pairs.graph_config(put_paths.PACK), f)
    ws_root = workspace_root or os.environ.get("PUT_WORKSPACE") or os.path.join(os.path.expanduser("~"), "put-ws", "runs")
    wt = os.path.join(ws_root, f"{os.path.basename(os.path.dirname(run_dir))}-{os.path.basename(run_dir)}")
    put_worktree.create(source, contract["base_sha"], wt)
    events_path = os.path.join(run_dir, "events.jsonl")
    try:
        card = put_tasks.card(task)
        cmd = command(card, files, cfg, contract["models"]["coder"], exe)
        session = run_session(cmd, wt, events_path, cfg["timeout_s"], cfg["output_budget_tokens"],
                              child_env("coder", version, label), popen=popen)
        with open(os.path.join(run_dir, "session.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(session, f, indent=1)
        diff = put_worktree.capture(wt)
        status = put_worktree.changed_paths(wt)
        with open(os.path.join(run_dir, "diff.patch"), "w", encoding="utf-8", newline="\n") as f:
            f.write(diff)
        with open(os.path.join(run_dir, "status.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(f"{s} {p}" for s, p in status) + "\n")
        snapshot(wt, status, run_dir)
        tests = evaluate_tests(wt, instance, task, diff, run_dir)
        with open(os.path.join(run_dir, "tests.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(tests, f, indent=1)
    finally:
        if not keep_worktree:
            put_worktree.remove(source, wt)
        shutil.rmtree(tmp, ignore_errors=True)
    return summarize(run_dir, task, body, rendered, contract, label)


def summarize(run_dir, task, body, rendered, contract, label):
    """Trace, answer and meta.json from what the run left on disk (also recovers a run whose summary failed)."""
    events_path = os.path.join(run_dir, "events.jsonl")
    with open(os.path.join(run_dir, "session.json"), encoding="utf-8") as f:
        session = json.load(f)
    events = erdos_phases.load(events_path)
    phases = erdos_phases.split(events)
    with open(os.path.join(run_dir, "trace.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(trace_md(events))
    with open(os.path.join(run_dir, "calls.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(calls_of(events), f, indent=1, ensure_ascii=False)
    with open(os.path.join(run_dir, "answer.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(phases.get("answer") or "")
    with open(os.path.join(run_dir, "status.txt"), encoding="utf-8") as f:
        changed = [line for line in f.read().splitlines() if line.strip()]
    meta = {"schema": 1, "task": task["id"], "split": task["split"], "rep": int(run_dir.rsplit(".r", 1)[-1]),
            "label": label, "prompt_version": put_prompt.version(body), "prompt_sha16": put_prompt.sha16(body),
            "rendered_checksum": put_prompt.checksum(rendered), "base_sha": contract["base_sha"],
            "session": session, "usage": usage_of(events), "marker": phases.get("marker"),
            "files_changed": len(changed), "events_sha256": sha256_file(events_path),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    return meta


def snapshot(wt, status, run_dir):
    """Copy every changed or added file as the agent left it to <run_dir>/after/<path>: the checks read the final
    text from here, so they can be recomputed from committed artifacts without the worktree."""
    for st, rel in status:
        if st.startswith("D"):
            continue
        src = os.path.join(wt, *rel.split("/"))
        if os.path.isfile(src):
            dst = os.path.join(run_dir, "after", *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)


def rebuild_after(run_dir, source, base_sha, work_root):
    """Recreate after/ for a run captured before snapshots existed: apply its diff to a fresh worktree."""
    wt = os.path.join(work_root, "rebuild-" + os.path.basename(run_dir))
    put_worktree.create(source, base_sha, wt)
    try:
        with open(os.path.join(run_dir, "diff.patch"), encoding="utf-8") as f:
            put_worktree.apply(wt, f.read())
        snapshot(wt, put_worktree.changed_paths(wt), run_dir)
    finally:
        put_worktree.remove(source, wt)


def evaluate_tests(wt, instance, task, diff, run_dir):
    """Compile the agent's tree, run its own new unit tests, then the hidden acceptance tests with pass-to-pass."""
    own = put_tasks.new_unit_tests(diff)
    out = {"own_classes": own}
    code, secs, _ = put_maven.compile_tests(wt, log_path=os.path.join(run_dir, "build.log"))
    out["build_green"] = code == 0
    out["compile_seconds"] = secs
    if code != 0:
        out.update({"own": [0, max(1, len(own))], "hidden": [0, 1], "pass_to_pass": [0, 1], "hidden_after_compile": False})
        return out
    if own:
        put_maven.test(wt, own, log_path=os.path.join(run_dir, "own-test.log"))
        rep = put_maven.reports(wt, own)
        out["own"] = list(put_maven.pass_rate(rep, own))
        out["own_reports"] = {k: {kk: v[kk] for kk in ("tests", "failures", "errors", "skipped")} for k, v in rep.items()}
    else:
        out["own"] = [0, 0]
    res = put_tasks.run_tests(wt, instance, task, os.path.join(run_dir, "acceptance"))
    out["hidden"] = res["hidden"]
    out["pass_to_pass"] = res["pass_to_pass"]
    out["hidden_after_compile"] = res["compiled"]
    out["acceptance_reports"] = res.get("reports", {})
    surefire = os.path.join(wt, "target", "surefire-reports")
    if os.path.isdir(surefire):
        shutil.copytree(surefire, os.path.join(run_dir, "surefire"), dirs_exist_ok=True)
    return out
