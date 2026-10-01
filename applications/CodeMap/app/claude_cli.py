"""The one `claude -p` runner every Claude role goes through (navigator, personas, judge, reflector).

Subscription-first law: the child never sees ANTHROPIC_API_KEY (a call that bills the API is a
bug); it authenticates through the user's login or CLAUDE_CODE_OAUTH_TOKEN from the environment
(`claude setup-token`). Every call is tagged with a role for telemetry, runs from a bare working
directory so the fixed context stays small, and returns the parsed JSON result with `usage`.
Stdlib only: this module shells out to the `claude` binary, never to an SDK (the terms of the
subscription permit the CLI; they do not permit third-party harnesses on the OAuth token).
"""

import json
import os
import shutil
import subprocess
import tempfile
import time

_STRIP = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
PROMPT_ARGV_MAX = 8000   # longer prompts are piped through stdin (see run)
DEFAULT_TIMEOUT = 240


def binary():
    return os.environ.get("CODEMAP_CLAUDE_BIN") or shutil.which("claude")


def available():
    return binary() is not None


def bare_cwd():
    """A fixed, empty directory: sessions persist under its slug, so --resume finds them."""
    d = os.environ.get("CODEMAP_CLI_CWD") or os.path.join(tempfile.gettempdir(), "codemap-cli")
    os.makedirs(d, exist_ok=True)
    return d


def child_env(role, persona=None, extra=None):
    env = {k: v for k, v in os.environ.items() if k not in _STRIP}
    attrs = [f"service.name=codemap", f"role={role}"]
    if persona:
        attrs.append(f"persona={persona}")
    if env.get("OTEL_RESOURCE_ATTRIBUTES"):
        attrs.append(env["OTEL_RESOURCE_ATTRIBUTES"])
    env["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(attrs)
    if extra:
        env.update(extra)
    return env


def build_cmd(prompt, model, system_file=None, mcp_config=None, allowed_tools=(), max_turns=1,
              resume=None, session_id=None, effort=None, json_schema=None, exe=None, append_system=False, tools=None):
    """append_system=False replaces Claude Code's own system prompt (the navigator: a smaller fixed
    context); True appends to it (personas working in a checkout keep the tool-use guidance)."""
    cmd = [exe or binary() or "claude", "-p", prompt, "--output-format", "json",
           "--model", model, "--max-turns", str(int(max_turns)), "--permission-mode", "dontAsk"]
    if system_file:
        cmd += ["--append-system-prompt-file" if append_system else "--system-prompt-file", system_file]
    if mcp_config is not None:
        cfg = mcp_config if isinstance(mcp_config, str) else json.dumps(mcp_config)
        cmd += ["--mcp-config", cfg, "--strict-mcp-config"]
    else:
        cmd += ["--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
    if allowed_tools:
        cmd += ["--allowedTools", ",".join(allowed_tools)]
    if tools is not None:  # the available tools; [] means none (a pure text answer, no tool turn to burn)
        cmd += ["--tools", ",".join(tools)]
    if resume:
        cmd += ["--resume", resume]
    elif session_id:
        cmd += ["--session-id", session_id]
    if effort:
        cmd += ["--effort", effort]
    if json_schema:
        cmd += ["--json-schema", json_schema if isinstance(json_schema, str) else json.dumps(json_schema)]
    return cmd


def parse_result(stdout):
    """The last JSON object on stdout is the result (warnings may precede it)."""
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        i = text.rfind("\n{")
        if i >= 0:
            try:
                return json.loads(text[i + 1:])
            except ValueError:
                return None
    return None


def run(prompt, model, role, system_file=None, mcp_config=None, allowed_tools=(), max_turns=1,
        resume=None, session_id=None, effort=None, json_schema=None, persona=None,
        timeout=DEFAULT_TIMEOUT, cwd=None, env_extra=None, runner=None, append_system=False, tools=None):
    """Run one `claude -p` call. Returns a dict that never raises: {text, usage, model_usage,
    session_id, is_error, error, num_turns, duration_ms, cost_usd, structured, raw}."""
    exe = binary()
    if exe is None and runner is None:
        return _fail("claude binary not found", model)
    cmd = build_cmd(prompt, model, system_file, mcp_config, allowed_tools, max_turns, resume,
                    session_id, effort, json_schema, exe=exe, append_system=append_system, tools=tools)
    env = child_env(role, persona, env_extra)
    t0 = time.time()
    try:
        if runner is not None:  # tests inject a fake subprocess
            proc = runner(cmd, env=env, cwd=cwd or bare_cwd(), timeout=timeout)
        else:
            stdin_text = None
            if len(prompt) > PROMPT_ARGV_MAX:  # Windows caps a command line near 32k characters: the prompt goes through stdin
                cmd = [c for c in cmd if c is not prompt]
                stdin_text = prompt
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",  # NOSONAR - argv list, no shell; the binary is Claude Code; see sonar-project.properties
                                  env=env, cwd=cwd or bare_cwd(), timeout=timeout, input=stdin_text)
    except subprocess.TimeoutExpired:
        return _fail(f"timeout after {timeout}s", model, duration_ms=int((time.time() - t0) * 1000))
    except OSError as ex:
        return _fail(f"launch failed: {ex}", model)
    res = parse_result(proc.stdout)
    duration_ms = int((time.time() - t0) * 1000)
    if res is None:
        err = (proc.stderr or "").strip()[-400:] or f"exit {proc.returncode} with no JSON on stdout"
        return _fail(err, model, duration_ms=duration_ms)
    usage = res.get("usage") or {}
    out = {
        "text": res.get("result") if isinstance(res.get("result"), str) else json.dumps(res.get("result")),
        "structured": res.get("structured_output"),
        "usage": {"input_tokens": int(usage.get("input_tokens", 0) or 0),
                  "output_tokens": int(usage.get("output_tokens", 0) or 0),
                  "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
                  "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0)},
        "model_usage": res.get("modelUsage") or {},
        "session_id": res.get("session_id"),
        "is_error": bool(res.get("is_error")),
        "error": (res.get("result") if res.get("is_error") else None),
        "num_turns": int(res.get("num_turns", 0) or 0),
        "duration_ms": int(res.get("duration_ms") or duration_ms),
        "cost_usd": res.get("total_cost_usd"),
        "model": model,
        "rate_limited": bool(res.get("is_error")) and "rate" in str(res.get("result", "")).lower(),
        "raw": res,
    }
    return out


def _fail(msg, model, duration_ms=0):
    return {"text": None, "structured": None, "usage": {"input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}, "model_usage": {},
            "session_id": None, "is_error": True, "error": msg, "num_turns": 0,
            "duration_ms": duration_ms, "cost_usd": None, "model": model, "rate_limited": False, "raw": None}
