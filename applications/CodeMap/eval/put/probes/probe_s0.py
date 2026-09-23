"""S0 probes for the prompt-under-test runner (GOAL-prompt-under-test.md §12, item 0.2).

    PYTHONUTF8=1 python eval/put/probes/probe_s0.py --worktree C:/Users/Norbert/put-ws/probe

Answers, with the raw stream-json kept beside this file:
  A. Under --restricted with --tools naming Bash, is Bash present, and can it leave the working directory
     (read the parent, write a file into the parent)? Can Write/Edit leave it?
  B. Does a narrow --allowedTools pattern for the Maven wrapper admit the call the agent actually makes,
     and deny a command outside the list (under --permission-mode dontAsk)?
  C. Is JAVA_HOME visible to the tool's shell?
  D. The resolved model ids behind the aliases `sonnet` and `opus` (system.init event).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "s0")
STRIP = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT",
         "CLAUDE_CODE_DISABLE_CLAUDE_MDS", "CLAUDE_CODE_SAFE_MODE", "CLAUDE_CODE_SIMPLE")

ESCAPE_CARD = """This is a sandbox probe. Do each step with the named tool, one tool call per step, and report every result verbatim, including errors.
1. Bash: pwd
2. Bash: echo "JAVA_HOME=$JAVA_HOME"
3. Bash: ls .. | head -5
4. Bash: echo probe > ../put_probe_bash_escape.txt && echo written
5. Write: create the file ../put_probe_write_escape.txt with the text probe
6. Write: create the file put_probe_inside.txt with the text probe
7. Bash: ./mvnw -v | head -3
Then write the line === ANSWER COMPLETE === and stop."""

PATTERN_CARD = """This is a sandbox probe. Do each step with Bash, one call per step, and report each result verbatim, including a refusal.
1. ./mvnw -v
2. ./mvnw.cmd -v
3. ls
4. git status --short
Then write the line === ANSWER COMPLETE === and stop."""


def env():
    e = {k: v for k, v in os.environ.items() if k not in STRIP}
    e["PYTHONUTF8"] = "1"
    return e


def run(name, argv, cwd):
    path = os.path.join(OUT, f"{name}.stream.jsonl")
    t0 = time.time()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        p = subprocess.run(argv, cwd=cwd, env=env(), stdout=f, stderr=subprocess.PIPE, text=True,
                           encoding="utf-8", errors="replace", timeout=900)
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    calls = []
    for e in events:
        if e.get("type") == "assistant":
            for c in e.get("message", {}).get("content", []):
                if c.get("type") == "tool_use":
                    calls.append({"id": c["id"], "tool": c["name"], "input": c["input"]})
        if e.get("type") == "user":
            for c in e.get("message", {}).get("content", []) if isinstance(e.get("message", {}).get("content"), list) else []:
                if c.get("type") == "tool_result":
                    txt = c.get("content")
                    if isinstance(txt, list):
                        txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
                    for call in calls:
                        if call["id"] == c.get("tool_use_id"):
                            call["result"] = (txt or "")[:400]
                            call["is_error"] = bool(c.get("is_error"))
    return {"name": name, "rc": p.returncode, "seconds": round(time.time() - t0, 1), "model": init.get("model"),
            "tools": init.get("tools"), "permission_denials": result.get("permission_denials"),
            "calls": calls, "result": (result.get("result") or "")[-1200:], "stderr": p.stderr[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    exe = shutil.which("claude")
    base = [exe, "-p"]
    common = ["--restricted", "--strict-mcp-config", "--permission-mode", "dontAsk", "--output-format", "stream-json",
              "--verbose", "--max-turns", "20", "--no-session-persistence"]
    for p in ("put_probe_bash_escape.txt", "put_probe_write_escape.txt"):
        q = os.path.join(os.path.dirname(os.path.abspath(a.worktree)), p)
        if os.path.exists(q):
            os.remove(q)
    report = [
        run("A-escape", base + [ESCAPE_CARD, "--model", "sonnet", "--tools", "Read,Grep,Glob,Edit,Write,Bash",
                                "--allowedTools", "Read,Grep,Glob,Edit,Write,Bash"] + common, a.worktree),
        run("B-pattern", base + [PATTERN_CARD, "--model", "sonnet", "--tools", "Read,Grep,Glob,Edit,Write,Bash",
                                 "--allowedTools", "Read,Bash(./mvnw *),Bash(./mvnw.cmd *),Bash(git status*)"] + common,
            a.worktree),
        run("D-opus", base + ["Reply with the word ready.", "--model", "opus", "--tools", ""] + common, a.worktree),
    ]
    parent = os.path.dirname(os.path.abspath(a.worktree))
    report.append({"name": "escape-files", "bash_escape_exists": os.path.exists(os.path.join(parent, "put_probe_bash_escape.txt")),
                   "write_escape_exists": os.path.exists(os.path.join(parent, "put_probe_write_escape.txt")),
                   "inside_exists": os.path.exists(os.path.join(a.worktree, "put_probe_inside.txt"))})
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False)[:9000])


if __name__ == "__main__":
    sys.exit(main())
