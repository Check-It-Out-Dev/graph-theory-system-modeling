"""One sorted JSON line per request — the artifact of record. Everything else is a view of it.

`emit` validates names and types against SCHEMA, appends under a lock (the server is threaded),
and never raises on disk trouble after validation (a telemetry failure must not fail an answer;
it is counted instead). `Meter` accumulates model usage across the calls one request makes.
"""

import json
import os
import threading
from datetime import datetime, timezone

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(R, "telemetry")
EVENT_TYPES = ("ask", "step", "open", "search", "feedback", "miss", "budget_refusal", "reload", "error")
TIERS = ("faq", "nav-sonnet", "nav-opus", "engine", "search", "none")
TERMINALS = ("answer", "abstain", "descend", "error", "budget_exhausted", "queue_full", "step", "ok")

# name -> allowed python types (None is always allowed for optional fields)
SCHEMA = {
    "schema": (int,), "event_type": (str,), "ts": (str,), "request_id": (str,),
    "parent_request_id": (str,), "session_id": (str,), "context_id": (str,), "user": (str,),
    "user_kind": (str,), "tier": (str,), "model": (str,), "model_version": (str,),
    "prompt_version": (str,), "pack_version": (str,), "protocol": (str,), "terminal": (str,),
    "steps": (int,), "duration_ms": (int,),
    "tokens": (dict,), "cache_read": (int,), "session_claude": (str,), "credits": (float, int),
    "faq_cache": (str,), "q": (str,), "answer": (str,), "pointers": (list,), "trajectory": (list,),
    "tool": (str,), "error": (str,), "rating": (int,), "vote": (str,), "tags": (list,),
    "comment": (str,), "verified": (bool,), "path": (str,), "why": (str,),
    "spent": (float, int), "budget": (float, int), "queue": (int,),
}
REQUIRED = ("schema", "event_type", "ts", "request_id", "user", "tier")
_LOCK = threading.Lock()
DROPPED = [0]


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def events_path(directory=None):
    d = directory or os.environ.get("CODEMAP_TELEMETRY_DIR") or DEFAULT_DIR
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "events.jsonl")


def validate(ev):
    for k in REQUIRED:
        if k not in ev:
            raise ValueError(f"event missing {k}")
    for k, v in ev.items():
        if k not in SCHEMA:
            raise ValueError(f"unknown event field {k}")
        if v is not None and not isinstance(v, SCHEMA[k]):
            raise ValueError(f"field {k} has type {type(v).__name__}")
    if ev["event_type"] not in EVENT_TYPES:
        raise ValueError(f"bad event_type {ev['event_type']}")
    if ev["tier"] not in TIERS:
        raise ValueError(f"bad tier {ev['tier']}")
    if ev.get("terminal") is not None and ev["terminal"] not in TERMINALS:
        raise ValueError(f"bad terminal {ev['terminal']}")
    if ev.get("answer") is not None and len(ev["answer"]) > 4000:
        ev["answer"] = ev["answer"][:4000]
    return ev


def emit(ev, directory=None, sink=None):
    """Validate, then append one sorted line. `sink` (a list) replaces the file in tests."""
    ev = dict(ev)
    ev.setdefault("schema", 1)
    ev = validate(ev)
    line = json.dumps(ev, sort_keys=True, ensure_ascii=False)
    if sink is not None:
        sink.append(ev)
        return ev
    try:
        with _LOCK:
            with open(events_path(directory), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:
        DROPPED[0] += 1
    return ev


def iter_events(directory=None):
    p = events_path(directory)
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


class Meter:
    """Sums Claude usage over the calls one request makes (Anthropic key names in, ours out)."""

    KEYS = ("prompt", "completion", "cached", "cache_creation")

    def __init__(self):
        self.prompt = self.completion = self.cached = self.cache_creation = 0
        self.calls = 0

    def add(self, usage):
        if not usage:
            return self
        self.prompt += int(usage.get("input_tokens", usage.get("prompt", 0)) or 0)
        self.completion += int(usage.get("output_tokens", usage.get("completion", 0)) or 0)
        self.cached += int(usage.get("cache_read_input_tokens", usage.get("cached", 0)) or 0)
        self.cache_creation += int(usage.get("cache_creation_input_tokens", usage.get("cache_creation", 0)) or 0)
        self.calls += 1
        return self

    def snapshot(self):
        return {k: getattr(self, k) for k in self.KEYS}
