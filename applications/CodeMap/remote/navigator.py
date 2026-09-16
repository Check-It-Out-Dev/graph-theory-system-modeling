"""The navigator tier: Claude Sonnet (Opus for `deep`) drives the engine and answers with pointers.

One `claude -p` per turn, system prompt = prompts/navigator/v<N>.md (L1 index, L2 prose, verb
table, curation notes), tools = the loopback engine MCP, `--resume` on follow-ups in the same
context. Concurrency is a semaphore (default 2) with a short queue: past it the caller gets
`queue_full` and a retry hint instead of a fourth process on a 4 GB box. Usage from the CLI's
JSON lands in the event; the trajectory comes from the engine MCP's trace file.
"""

import json
import os
import re
import sys
import tempfile
import threading
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(R, "app"))

import claude_cli  # noqa: E402

from . import contexts, pointers, telemetry, tools as tools_mod  # noqa: E402

MODELS = {"auto": os.environ.get("CODEMAP_NAV_MODEL", "claude-sonnet-5"),
          "deep": os.environ.get("CODEMAP_NAV_MODEL_DEEP", "claude-opus-5")}
TIER_OF = {"auto": "nav-sonnet", "deep": "nav-opus"}
_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


class Navigator:
    def __init__(self, app, max_concurrent=None, queue_max=None, max_turns=None, timeout=None, effort=None):
        self.app = app
        self.max_concurrent = int(max_concurrent or os.environ.get("CODEMAP_NAV_CONCURRENCY", "2"))
        self.queue_max = int(queue_max or os.environ.get("CODEMAP_NAV_QUEUE", "8"))
        self.max_turns = int(max_turns or os.environ.get("CODEMAP_NAV_TURNS", "10"))
        self.timeout = int(timeout or os.environ.get("CODEMAP_NAV_TIMEOUT", "240"))
        self.effort = effort or os.environ.get("CODEMAP_NAV_EFFORT", "medium")
        self.contexts = contexts.Contexts()
        self._sem = threading.BoundedSemaphore(self.max_concurrent)
        self._waiting = 0
        self._lock = threading.Lock()
        self.runner = None  # tests inject a fake subprocess

    def describe(self):
        return {"models": MODELS, "max_concurrent": self.max_concurrent, "queue_max": self.queue_max,
                "max_turns": self.max_turns, "effort": self.effort, "waiting": self._waiting,
                "contexts": len(self.contexts), "claude": claude_cli.available()}

    # ------------------------------------------------------------------ mcp config
    def mcp_config(self, trace_file):
        env = {"CODEMAP_TRACE_FILE": trace_file, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               "CODEMAP_PACK_DIR": self.app.pack_dir}
        # absolute script path: Claude Code does not honour a cwd in the MCP config, and the bare
        # working directory of the navigator has no `remote` package on its path
        return {"mcpServers": {"engine": {"command": sys.executable,
                                          "args": [os.path.join(HERE, "engine_mcp.py")], "env": env}}}

    # ------------------------------------------------------------------ ask
    def ask(self, app, ident, q, context_id, tier, faq_hit):
        tier = tier if tier in MODELS else "auto"
        model = MODELS[tier]
        tname = TIER_OF[tier]
        st = app.budget_state(ident.user) if hasattr(app, "budget_state") else {"exhausted": False}
        if st.get("exhausted"):
            ev = tools_mod.base_event(app, ident, "budget_refusal", tname, context_id=context_id, protocol="navigator",
                                      terminal="budget_exhausted", q=q, credits=0.0, steps=0, model=model,
                                      tool="codemap_ask", spent=st["spent"], budget=st["budget"])
            app.emit(ev)
            out = {"error": "budget_exhausted", "terminal": "budget_exhausted", "request_id": ident.request_id,
                   "context_id": context_id, "spent": st["spent"], "budget": st["budget"],
                   "retry_after_s": st.get("resets_in_s"), "note": "daily credits are spent; the engine tools stay free"}
            return json.dumps(out), True
        try:
            ctx, created = self.contexts.get_or_create(context_id, ident.user["id"])
        except PermissionError as ex:
            return self._refuse(app, ident, q, context_id, tname, model, "error", str(ex))
        with self._lock:
            if self._waiting >= self.queue_max:
                return self._refuse(app, ident, q, context_id, tname, model, "queue_full",
                                    f"navigator busy ({self._waiting} waiting); retry in 30 s", queue=self._waiting)
            self._waiting += 1
        try:
            with self._sem:
                with self._lock:
                    self._waiting -= 1
                return self._run(app, ident, q, ctx, created, tier, model, tname, faq_hit)
        finally:
            pass

    def _run(self, app, ident, q, ctx, created, tier, model, tname, faq_hit):
        trace = os.path.join(tempfile.gettempdir(), f"codemap-trace-{ident.request_id}.jsonl")
        prompt = q if not ctx.turns else f"Follow-up in the same conversation: {q}"
        nearest = faq_hit.get("nearest") if isinstance(faq_hit, dict) else None
        if nearest and created:
            prompt += f"\n\n(An FAQ entry is nearby but did not match: {json.dumps(nearest)[:400]})"
        resume = ctx.session_claude if not created and ctx.session_claude else None
        session_id = None if resume else str(uuid.uuid4())
        res = claude_cli.run(prompt, model, role="navigator", system_file=app.prompt_path,
                             mcp_config=self.mcp_config(trace), allowed_tools=("mcp__engine__*",),
                             max_turns=self.max_turns, resume=resume, session_id=session_id,
                             effort=self.effort, persona=ident.user["id"], timeout=self.timeout,
                             runner=self.runner)
        trajectory = _read_trace(trace)
        meter = telemetry.Meter().add(res["usage"])
        if res["is_error"]:
            terminal, answer, ptr_names = "error", None, []
        else:
            answer, ptr_names, terminal = parse_answer(res["text"])
            if res.get("session_id"):
                ctx.session_claude = res["session_id"]
        ptrs = [pointers.to_pointer(app.entity(n)) for n in ptr_names if app.entity(n)]
        if not ptrs and answer:
            ptrs = [pointers.to_pointer(app.entity(n)) for n in app.names_in(answer)][:8]
        credits = app.credits_for(tname, model, meter.snapshot()) if hasattr(app, "credits_for") else 0.0
        ev = tools_mod.base_event(app, ident, "ask", tname, context_id=ctx.context_id, protocol="navigator",
                                  terminal=terminal, faq_cache="miss", q=q, answer=answer, pointers=ptrs,
                                  credits=credits, steps=len(trajectory), tokens=meter.snapshot(),
                                  cache_read=meter.cached, model=model, model_version=model,
                                  session_claude=res.get("session_id"), trajectory=[t["arg"] for t in trajectory][:40],
                                  tool="codemap_ask", error=(res.get("error") or None) if res["is_error"] else None)
        ev["duration_ms"] = ident.ms()
        app.emit(ev)
        ctx.turns.append({"request_id": ident.request_id, "ts": ev["ts"], "q": q, "tier": tname,
                          "terminal": terminal, "credits": credits})
        out = {"answer": answer, "pointers": ptrs, "request_id": ident.request_id, "context_id": ctx.context_id,
               "tier": tname, "model": model, "terminal": terminal, "credits": credits,
               "steps": len(trajectory), "tokens": meter.snapshot(), "cache_read": meter.cached,
               "turn": len(ctx.turns), "note": "verify at least one pointer in your checkout, then codemap_feedback"}
        if res["is_error"]:
            out["error"] = res.get("error")
            if res.get("rate_limited"):
                out["retry_after_s"] = 900
        return json.dumps(out, ensure_ascii=False), bool(res["is_error"])

    def _refuse(self, app, ident, q, context_id, tname, model, terminal, msg, queue=None):
        app.emit(tools_mod.base_event(app, ident, "ask", tname, context_id=context_id, protocol="navigator",
                                      terminal=terminal, faq_cache="miss", q=q, credits=0.0, steps=0,
                                      model=model, tool="codemap_ask", error=msg, queue=queue))
        out = {"error": msg, "terminal": terminal, "request_id": ident.request_id, "context_id": context_id}
        if terminal == "queue_full":
            out["retry_after_s"] = 30
        return json.dumps(out), True


def parse_answer(text):
    """Prose + a fenced json block {terminal, pointers[]} -> (prose, [names], terminal)."""
    text = text or ""
    m = None
    for m in _JSON_BLOCK.finditer(text):
        pass
    if m:
        try:
            meta = json.loads(m.group(1))
            prose = (text[:m.start()] + text[m.end():]).strip()
            names = [str(n) for n in (meta.get("pointers") or []) if isinstance(n, (str, int))]
            terminal = "abstain" if str(meta.get("terminal", "answer")).lower().startswith("abst") else "answer"
            return prose, names, terminal
        except ValueError:
            pass
    low = text.lower()
    terminal = "abstain" if ("cannot answer" in low or "not in this codebase" in low or "out of corpus" in low) else "answer"
    return text.strip(), [], terminal


def _read_trace(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    try:
        os.remove(path)
    except OSError:
        pass
    return out
