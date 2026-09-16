"""Tool bodies. Each takes the App (engine + registries) and an Ident (who is calling) and returns
(text, is_error); the server emits exactly one event per call from what they return.

S1 ships ask (FAQ or descend), step, open, status; search / feedback / miss answer "not yet" until
their slices land (S4). The navigator tier replaces `descend` in S2.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(R, "app"))

from dsl import execute, ParseError  # noqa: E402
from engine import DslError  # noqa: E402

from . import ids, pointers, telemetry  # noqa: E402


class Ident:
    __slots__ = ("user", "session_id", "request_id", "started")

    def __init__(self, user, session_id=None):
        self.user = user
        self.session_id = session_id
        self.request_id = ids.new_request_id()
        self.started = time.time()

    def ms(self):
        return int((time.time() - self.started) * 1000)


def _text(obj):
    return json.dumps(obj, ensure_ascii=False)


def base_event(app, ident, event_type, tier, **extra):
    ev = {"schema": 1, "event_type": event_type, "ts": telemetry.now_iso(),
          "request_id": ident.request_id, "session_id": ident.session_id,
          "user": ident.user["id"], "user_kind": ident.user["kind"], "tier": tier,
          "pack_version": app.pack_version, "prompt_version": app.prompt_version,
          "duration_ms": ident.ms()}
    ev.update(extra)
    return ev


# ----------------------------------------------------------------------------- ask

def ask(app, ident, args):
    q = (args.get("q") or "").strip()
    if not q:
        return _text({"error": "empty question"}), True
    context_id = args.get("context_id") or ids.new_request_id()
    nav = getattr(app, "navigator", None)
    # a follow-up inside a conversation is never an FAQ question: "and what depends on it?" needs the context
    follow_up = bool(nav and nav.contexts.get(context_id) and nav.contexts.get(context_id).turns)
    hit = {"kind": "skipped"} if follow_up else app.engine.cache(q)
    if hit.get("kind") == "cache_hit":
        ptrs = _pointers_from_gold(app, hit)
        out = {"answer": hit["answer"], "pointers": ptrs, "request_id": ident.request_id,
               "context_id": context_id, "tier": "faq", "terminal": "answer", "credits": 0.0,
               "faq_id": hit.get("id"), "note": "answered from the curated FAQ (0 credits); rate it too"}
        app.emit(base_event(app, ident, "ask", "faq", context_id=context_id, protocol="cache",
                            terminal="answer", faq_cache="hit", q=q, answer=hit["answer"],
                            pointers=ptrs, credits=0.0, steps=0, tool="codemap_ask"))
        return _text(out), False
    if nav is not None:
        return nav.ask(app, ident, q, context_id, args.get("tier") or "auto", hit)
    l1 = app.engine.map()
    out = {"answer": None, "terminal": "descend", "request_id": ident.request_id, "context_id": context_id,
           "tier": "none", "credits": 0.0,
           "l1_index": l1.get("index"), "caveats": l1.get("caveats"),
           "note": "no navigator configured on this server; use codemap_step to navigate the L1 index"}
    app.emit(base_event(app, ident, "ask", "none", context_id=context_id, protocol="descend",
                        terminal="descend", faq_cache="miss", q=q, credits=0.0, steps=0, tool="codemap_ask"))
    return _text(out), False


def _pointers_from_gold(app, hit):
    ptrs = []
    for row in hit.get("gold_result_excerpt") or []:
        name = row[0] if isinstance(row, (list, tuple)) and row else None
        ent = app.entity(name) if name else None
        if ent:
            ptrs.append(pointers.to_pointer(ent))
    if not ptrs:
        # fall back to entity names mentioned in the answer text
        for name in app.names_in(hit.get("answer", "")):
            ptrs.append(pointers.to_pointer(app.entity(name)))
    return ptrs[:8]


# ----------------------------------------------------------------------------- step

def step(app, ident, args):
    dsl = (args.get("dsl") or "").strip()
    try:
        res = execute(app.engine, dsl)
    except (ParseError, DslError) as e:
        app.emit(base_event(app, ident, "step", "engine", protocol="step", terminal="error",
                            error=str(e)[:300], credits=0.0, steps=1, tool="codemap_step", q=dsl))
        return _text({"error": str(e), "verbs": "map enter find impact flow seam cohort spine health read"}), True
    if res.get("kind") == "pointer":
        ent = app.entity(res.get("file"))
        if ent:
            res["pointer"] = pointers.to_pointer(ent)
            res.pop("path", None)  # never leak the authoring box path
    app.emit(base_event(app, ident, "step", "engine", protocol="step", terminal="step",
                        credits=0.0, steps=1, tool="codemap_step", q=dsl))
    return _text(res), False


# ----------------------------------------------------------------------------- open

def open_(app, ident, args):
    name = (args.get("name") or "").strip()
    ent = app.entity(name)
    if not ent:
        cands = [e["name"] for e in app.engine.ents if name.lower() in e["name"].lower()][:10]
        app.emit(base_event(app, ident, "open", "engine", protocol="step", terminal="error",
                            credits=0.0, tool="codemap_open", q=name, error="unknown entity"))
        return _text({"error": f"unknown entity {name!r}", "candidates": cands}), True
    sub = ent.get("curated") or ent.get("subsystem")
    clue = app.clue(sub)
    ptr = pointers.to_pointer(ent, clue=clue)
    app.emit(base_event(app, ident, "open", "engine", protocol="step", terminal="ok",
                        credits=0.0, tool="codemap_open", q=name, pointers=[ptr]))
    return _text(ptr), False


# ----------------------------------------------------------------------------- status

def status(app, ident, args):
    st = app.status()
    st["you"] = {"user": ident.user["id"], "kind": ident.user["kind"],
                 "daily_credit_budget": ident.user["daily_credit_budget"],
                 "spent_today": app.spent_today(ident.user["id"]),
                 "remaining": max(0.0, ident.user["daily_credit_budget"] - app.spent_today(ident.user["id"]))}
    return _text(st), False


# ----------------------------------------------------------------------------- not yet

def _not_yet(which):
    def f(app, ident, args):
        return _text({"error": f"{which} is not available on this server yet"}), True
    return f


HANDLERS = {
    "codemap_ask": ask,
    "codemap_step": step,
    "codemap_open": open_,
    "codemap_status": status,
    "codemap_search": _not_yet("codemap_search"),
    "codemap_feedback": _not_yet("codemap_feedback"),
    "codemap_miss": _not_yet("codemap_miss"),
}
