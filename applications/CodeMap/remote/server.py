"""CodeMap Remote — the served MCP. Stdlib http.server + the v1 engine over the Release pack.

    CODEMAP_TOKEN=... python -m remote.server [--bind 0.0.0.0 --port 7345]

Routes: POST /mcp (Streamable HTTP JSON-RPC) · GET /healthz · GET /status · GET /users ·
GET /budget?user= · GET /metrics (S3) · POST /feedback (S4) · POST /admin/reload (S6).
Auth: `Authorization: Bearer $CODEMAP_TOKEN` (401) and `X-CodeMap-User` from users.json (400).
One shared token, names from an enum: a small-team deployment by design (README says so).
Handlers are pure functions `(body, headers) -> (code, obj)`; the HTTP class only dispatches.
"""

import argparse
import hmac
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "app"))

import claude_cli  # noqa: E402
from engine import Engine  # noqa: E402

from remote import credits, feedback, ids, mcp, metrics, reload as reload_mod, search, telemetry, telemetry_push, tools, users  # noqa: E402

VERSION = "1.2.0"


class App:
    """Everything a handler needs: engine, users, versions, event sink, per-user spend."""

    def __init__(self, engine=None, users_map=None, token=None, admin_token=None,
                 pack_dir=None, prompt_path=None, sink=None, navigator=None):
        self.engine = engine or Engine()
        self.users = users_map or users.load()
        self.token = token
        self.admin_token = admin_token
        self.pack_dir = pack_dir or os.environ.get("CODEMAP_PACK_DIR") or os.path.join(R, "graph", "pack")
        self.pack_version = ids.pack_version(self.pack_dir)
        # the active prompt is built from template + pack + notes unless a file is pinned (tests, GEPA candidates)
        self.prompt_path = prompt_path or os.environ.get("CODEMAP_NAV_PROMPT")
        if not self.prompt_path:
            active = os.path.join(os.environ.get("CODEMAP_TELEMETRY_DIR") or telemetry.DEFAULT_DIR, "navigator-active.md")
            try:
                self.prompt_path = reload_mod.build_active_prompt(self, active)
            except Exception:  # a broken pack must not stop the server from answering FAQ/engine calls
                self.prompt_path = os.path.join(R, "prompts", "navigator", "v1.md")
        self.prompt_version = ids.prompt_version(self.prompt_path)
        self.poller = None
        self.reloads = []
        self.sink = sink  # list in tests, None in production (file)
        self.navigator = None
        self._by_name = {e["name"]: e for e in self.engine.ents}
        self._lock = threading.Lock()
        self.started = telemetry.now_iso()
        self.card = credits.RateCard()
        self.ledger = credits.Ledger()
        self.registry = metrics.Registry()
        self.seen = feedback.Seen()
        self.pusher = None
        self.replayed = 0
        self.backlog_path = os.environ.get("CODEMAP_BACKLOG") or os.path.join(R, "graph", "delta", "backlog.jsonl")
        self.backlog_lock = threading.Lock()
        if sink is None:  # production: the views are rebuilt from the artifact of record
            for ev in telemetry.iter_events():
                self.ledger.observe(ev)
                self.registry.observe(ev)
                self.seen.observe(ev)
                self.replayed += 1
        self.search_index = None
        if os.environ.get("CODEMAP_SEARCH", "auto") != "off" and sink is None:
            self.search_index = search.SearchIndex(self)
            if not self.search_index.load():
                self.search_index.build_async()
        self.registry.set("codemap_info", {"version": VERSION, "pack_version": self.pack_version,
                                           "prompt_version": self.prompt_version}, 1)
        for u in self.users.values():
            self.registry.set("codemap_budget_remaining", {"user": u["id"]}, self.ledger.state(u)["remaining"])
        if navigator is None:
            mode = os.environ.get("CODEMAP_NAVIGATOR", "auto")
            if mode != "off" and claude_cli.available() and os.path.exists(self.prompt_path):
                from remote.navigator import Navigator
                navigator = Navigator(self)
        self.navigator = navigator or None

    # --- event + spend -------------------------------------------------------------
    def emit(self, ev):
        ev = telemetry.emit(ev, sink=self.sink)
        self.ledger.observe(ev)
        self.registry.observe(ev)
        self.seen.observe(ev)
        u = self.users.get(ev.get("user"))
        if u:
            self.registry.set("codemap_budget_remaining", {"user": u["id"]}, self.ledger.state(u)["remaining"])
        if self.pusher:
            self.pusher.on_event(ev)
        return ev

    def spent_today(self, user_id):
        return self.ledger.spent(user_id)

    def budget_state(self, user_doc):
        return self.ledger.state(user_doc)

    def credits_for(self, tier, model, tokens, flat=None):
        return self.card.compute(tier, tokens, flat)

    # --- pack helpers ---------------------------------------------------------------
    def entity(self, name):
        return self._by_name.get(name)

    def names_in(self, text):
        return [n for n in self._by_name if n in (text or "")]

    def clue_full(self, sub):
        l2 = self.engine.l2 if isinstance(self.engine.l2, dict) else {}
        return l2.get(str(sub)) or l2.get(_int(sub))

    def clue(self, sub):
        nav = self.clue_full(sub)
        if not nav:
            return None
        return {"subsystem": nav.get("sub_id", sub), "name": nav.get("name"), "role": nav.get("role"),
                "summary": nav.get("ai_summary")}

    # --- reload -----------------------------------------------------------------------
    def reload(self, fetch=True, runner=None):
        """Fetch the latest pack (unless fetch=False), rebuild engine, prompt, indexes. -> (ok, detail)"""
        detail = ""
        if fetch:
            ok, detail = reload_mod.fetch_latest(self.pack_dir, runner)
            if not ok:
                self.reloads.append({"at": telemetry.now_iso(), "ok": False, "detail": detail[-300:]})
                return False, detail
        try:
            engine = Engine()
        except Exception as ex:
            self.reloads.append({"at": telemetry.now_iso(), "ok": False, "detail": f"engine: {ex}"[:300]})
            return False, f"engine failed to open the new pack: {ex}"
        with self._lock:
            self.engine = engine
            self._by_name = {e["name"]: e for e in engine.ents}
            old_pack, old_prompt = self.pack_version, self.prompt_version
            self.pack_version = ids.pack_version(self.pack_dir)
            if not os.environ.get("CODEMAP_NAV_PROMPT"):
                try:
                    active = os.path.join(os.environ.get("CODEMAP_TELEMETRY_DIR") or telemetry.DEFAULT_DIR, "navigator-active.md")
                    self.prompt_path = reload_mod.build_active_prompt(self, active)
                except Exception:
                    pass
            self.prompt_version = ids.prompt_version(self.prompt_path)
            if self.navigator is not None:
                self.navigator.contexts = __import__("remote.contexts", fromlist=["Contexts"]).Contexts()
            if self.search_index is not None:
                self.search_index = search.SearchIndex(self)
                if not self.search_index.load():
                    self.search_index.build_async()
        self.registry.set("codemap_info", {"version": VERSION, "pack_version": self.pack_version,
                                           "prompt_version": self.prompt_version}, 1)
        ev = {"schema": 1, "event_type": "reload", "ts": telemetry.now_iso(), "request_id": ids.new_request_id(),
              "user": "ci", "user_kind": "system", "tier": "none", "credits": 0.0, "terminal": "ok",
              "pack_version": self.pack_version, "prompt_version": self.prompt_version,
              "comment": f"pack {old_pack} -> {self.pack_version}; prompt {old_prompt} -> {self.prompt_version}"}
        self.emit(ev)
        self.reloads.append({"at": ev["ts"], "ok": True, "detail": ev["comment"]})
        return True, ev["comment"]

    def status(self):
        return {"service": "codemap-remote", "version": VERSION, "started": self.started,
                "pack_version": self.pack_version, "prompt_version": self.prompt_version,
                "entities": len(self.engine.ents), "navigators": len(self.engine.l2),
                "ladybug": bool(self.engine.lb), "users": sorted(self.users),
                "navigator": self.navigator.describe() if self.navigator else None,
                "events_dropped": telemetry.DROPPED[0], "events_replayed": self.replayed,
                "answers_seen": len(self.seen), "reloads": self.reloads[-5:],
                "poll": (self.poller.last if self.poller else None),
                "search": ({"state": self.search_index.state, "entities": len(self.search_index.names),
                            "error": self.search_index.error} if self.search_index else None),
                "push": (self.pusher.stats if self.pusher else None),
                "rate_card": self.card.per_1k}

    # --- auth -----------------------------------------------------------------------
    def authed(self, headers):
        if not self.token:
            return True  # local mode: no token configured
        got = _hdr(headers, "Authorization") or ""
        if not got.lower().startswith("bearer "):
            return False
        return hmac.compare_digest(got[7:].strip(), self.token)

    def is_admin(self, headers):
        if not self.admin_token:
            return False
        got = _hdr(headers, "X-CodeMap-Admin") or ""
        return hmac.compare_digest(got.strip(), self.admin_token)


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _hdr(headers, name):
    if headers is None:
        return None
    v = headers.get(name)
    if v is None and hasattr(headers, "get"):
        v = headers.get(name.lower())
    return v


# --------------------------------------------------------------------------- handlers

def handle_mcp(app, body, headers):
    if not app.authed(headers):
        return 401, {"error": "unauthorized"}
    user, wanted = users.resolve(app.users, headers, body if isinstance(body, dict) else None)
    if user is None:
        return 400, {"error": "unknown_user", "wanted": wanted, "known": sorted(app.users)}
    session_id = _hdr(headers, "Mcp-Session-Id") or _hdr(headers, "X-CodeMap-Session")

    def call_tool(name, args):
        ident = tools.Ident(user, session_id)
        return tools.HANDLERS[name](app, ident, args)

    return mcp.dispatch_body(body, call_tool)


def handle_healthz(app):
    return 200, {"ok": True, "pack_version": app.pack_version, "prompt_version": app.prompt_version}


def handle_status(app, headers):
    if not app.authed(headers):
        return 401, {"error": "unauthorized"}
    return 200, app.status()


def handle_users(app, headers):
    if not app.authed(headers):
        return 401, {"error": "unauthorized"}
    return 200, {"users": [{"id": u["id"], "display_name": u["display_name"], "kind": u["kind"],
                            "daily_credit_budget": u["daily_credit_budget"],
                            "spent_today": app.spent_today(u["id"])} for u in app.users.values()]}


def handle_budget(app, headers, query):
    if not app.authed(headers):
        return 401, {"error": "unauthorized"}
    uid = (query.get("user") or [""])[0] or _hdr(headers, users.HEADER) or "anonymous"
    u = app.users.get(uid)
    if not u:
        return 400, {"error": "unknown_user", "wanted": uid, "known": sorted(app.users)}
    st = app.budget_state(u)
    return (429 if st["exhausted"] else 200), st


def handle_metrics(app):
    return 200, app.registry.render()


def handle_reload(app, headers):
    if not app.authed(headers) or not app.is_admin(headers):
        return 401, {"error": "unauthorized"}
    ok, detail = app.reload()
    return (200 if ok else 502), {"ok": ok, "detail": detail, "pack_version": app.pack_version,
                                   "prompt_version": app.prompt_version}


def handle_events(app, headers, query):
    """Export event lines (admin): the judge and the quality run read the artifact of record from here."""
    if not app.authed(headers) or not app.is_admin(headers):
        return 401, {"error": "unauthorized"}
    since = (query.get("since") or [""])[0]
    limit = int((query.get("limit") or ["5000"])[0])
    out = []
    for ev in telemetry.iter_events():
        if since and (ev.get("ts") or "") < since:
            continue
        out.append(ev)
    return 200, {"events": out[-limit:], "count": len(out), "since": since or None}


def handle_backlog(app, headers):
    """The saturation backlog (codemap_miss rows) for the night runner to sync into the repo."""
    if not app.authed(headers) or not app.is_admin(headers):
        return 401, {"error": "unauthorized"}
    rows = []
    try:
        with open(app.backlog_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return 200, {"rows": rows[-2000:], "count": len(rows)}


def handle_feedback(app, body, headers):
    """REST twin of the codemap_feedback tool (the UI and curl use it)."""
    if not app.authed(headers):
        return 401, {"error": "unauthorized"}
    user, wanted = users.resolve(app.users, headers, body if isinstance(body, dict) else None)
    if user is None:
        return 400, {"error": "unknown_user", "wanted": wanted, "known": sorted(app.users)}
    ident = tools.Ident(user, _hdr(headers, "X-CodeMap-Session"))
    text, is_error = feedback.feedback(app, ident, body if isinstance(body, dict) else {})
    obj = json.loads(text)
    if is_error:
        code = 404 if "unknown request_id" in obj.get("error", "") else 400
        return code, obj
    return 201, obj


# --------------------------------------------------------------------------- HTTP

class H(BaseHTTPRequestHandler):
    app = None  # set by serve()
    server_version = f"codemap-remote/{VERSION}"

    def log_message(self, *a):  # quiet; events are the log
        pass

    def _send(self, code, obj, extra=None):
        b = b"" if obj is None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if b:
            self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/healthz":
            return self._send(*handle_healthz(self.app))
        if u.path == "/status":
            return self._send(*handle_status(self.app, self.headers))
        if u.path == "/users":
            return self._send(*handle_users(self.app, self.headers))
        if u.path == "/budget":
            code, st = handle_budget(self.app, self.headers, parse_qs(u.query))
            extra = {"Retry-After": str(st.get("resets_in_s", 0))} if code == 429 else None
            return self._send(code, st, extra)
        if u.path == "/metrics":
            code, text = handle_metrics(self.app)
            b = text.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            return self.wfile.write(b)
        if u.path == "/admin/backlog":
            return self._send(*handle_backlog(self.app, self.headers))
        if u.path == "/admin/events":
            return self._send(*handle_events(self.app, self.headers, parse_qs(u.query)))
        if u.path == "/mcp":
            return self._send(405, {"error": "no server-initiated stream; POST JSON-RPC to /mcp"})
        return self._send(404, {"error": "not found"})

    def do_DELETE(self):
        if urlparse(self.path).path == "/mcp":
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            return self._send(400, {"error": "bad json"})
        if u.path == "/mcp":
            code, reply = handle_mcp(self.app, body, self.headers)
            return self._send(code, reply, {"Mcp-Session-Id": self.headers.get("Mcp-Session-Id") or ids.new_request_id()})
        if u.path == "/feedback":
            return self._send(*handle_feedback(self.app, body, self.headers))
        if u.path == "/admin/reload":
            return self._send(*handle_reload(self.app, self.headers))
        return self._send(404, {"error": "unknown endpoint"})


def serve(bind="127.0.0.1", port=7345, app=None):
    app = app or App(token=os.environ.get("CODEMAP_TOKEN") or None,
                     admin_token=os.environ.get("CODEMAP_ADMIN_TOKEN") or None)
    H.app = app
    if not app.token and bind not in ("127.0.0.1", "localhost"):
        raise SystemExit("refusing to bind a non-loopback address without CODEMAP_TOKEN")
    app.pusher = telemetry_push.Pusher(app.registry)
    pushing = app.pusher.start()
    app.poller = reload_mod.Poller(app)
    polling = app.poller.start()
    from remote import housekeeping
    pruned = housekeeping.prune_sessions()
    print(f"codemap-remote {VERSION}: {len(app.engine.ents)} entities, pack {app.pack_version}, "
          f"prompt {app.prompt_version}, token={'set' if app.token else 'NONE (local)'}, "
          f"replayed {app.replayed} events, push={'on' if pushing else 'off'}, pack-poll={'on' if polling else 'off'}, "
          f"sessions pruned {pruned[0]} -> http://{bind}:{port}/mcp")
    try:
        ThreadingHTTPServer((bind, port), H).serve_forever()  # NOSONAR - bind is configured; loopback by default; see sonar-project.properties
    finally:
        app.pusher.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default=os.environ.get("CODEMAP_BIND", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CODEMAP_PORT", "7345")))
    a = ap.parse_args()
    serve(a.bind, a.port)


if __name__ == "__main__":
    main()
