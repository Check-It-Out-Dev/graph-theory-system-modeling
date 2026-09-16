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

from engine import Engine  # noqa: E402

from remote import ids, mcp, telemetry, tools, users  # noqa: E402

VERSION = "1.2.0"


class App:
    """Everything a handler needs: engine, users, versions, event sink, per-user spend."""

    def __init__(self, engine=None, users_map=None, token=None, admin_token=None,
                 pack_dir=None, prompt_path=None, sink=None):
        self.engine = engine or Engine()
        self.users = users_map or users.load()
        self.token = token
        self.admin_token = admin_token
        self.pack_dir = pack_dir or os.environ.get("CODEMAP_PACK_DIR") or os.path.join(R, "graph", "pack")
        self.prompt_path = prompt_path or os.environ.get("CODEMAP_NAV_PROMPT") or os.path.join(R, "prompts", "navigator", "v1.md")
        self.pack_version = ids.pack_version(self.pack_dir)
        self.prompt_version = ids.prompt_version(self.prompt_path)
        self.sink = sink  # list in tests, None in production (file)
        self.navigator = None  # set in S2
        self._by_name = {e["name"]: e for e in self.engine.ents}
        self._lock = threading.Lock()
        self._spent = {}
        self.started = telemetry.now_iso()

    # --- event + spend -------------------------------------------------------------
    def emit(self, ev):
        ev = telemetry.emit(ev, sink=self.sink)
        c = ev.get("credits") or 0
        if c:
            with self._lock:
                self._spent[ev["user"]] = self._spent.get(ev["user"], 0.0) + float(c)
        return ev

    def spent_today(self, user_id):
        return round(self._spent.get(user_id, 0.0), 4)

    # --- pack helpers ---------------------------------------------------------------
    def entity(self, name):
        return self._by_name.get(name)

    def names_in(self, text):
        return [n for n in self._by_name if n in (text or "")]

    def clue(self, sub):
        l2 = self.engine.l2 if isinstance(self.engine.l2, dict) else {}
        nav = l2.get(str(sub)) or l2.get(_int(sub))
        if not nav:
            return None
        return {"subsystem": nav.get("sub_id", sub), "name": nav.get("name"), "role": nav.get("role"),
                "summary": nav.get("ai_summary")}

    def status(self):
        return {"service": "codemap-remote", "version": VERSION, "started": self.started,
                "pack_version": self.pack_version, "prompt_version": self.prompt_version,
                "entities": len(self.engine.ents), "navigators": len(self.engine.l2),
                "ladybug": bool(self.engine.lb), "users": sorted(self.users),
                "navigator": self.navigator.describe() if self.navigator else None,
                "events_dropped": telemetry.DROPPED[0]}

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
    spent = app.spent_today(uid)
    return 200, {"user": uid, "budget": u["daily_credit_budget"], "spent": spent,
                 "remaining": round(max(0.0, u["daily_credit_budget"] - spent), 4)}


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
            return self._send(*handle_budget(self.app, self.headers, parse_qs(u.query)))
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
        return self._send(404, {"error": "unknown endpoint"})


def serve(bind="127.0.0.1", port=7345, app=None):
    app = app or App(token=os.environ.get("CODEMAP_TOKEN") or None,
                     admin_token=os.environ.get("CODEMAP_ADMIN_TOKEN") or None)
    H.app = app
    if not app.token and bind not in ("127.0.0.1", "localhost"):
        raise SystemExit("refusing to bind a non-loopback address without CODEMAP_TOKEN")
    print(f"codemap-remote {VERSION}: {len(app.engine.ents)} entities, pack {app.pack_version}, "
          f"prompt {app.prompt_version}, token={'set' if app.token else 'NONE (local)'} -> http://{bind}:{port}/mcp")
    ThreadingHTTPServer((bind, port), H).serve_forever()  # NOSONAR - bind is configured; loopback by default; see sonar-project.properties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default=os.environ.get("CODEMAP_BIND", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CODEMAP_PORT", "7345")))
    a = ap.parse_args()
    serve(a.bind, a.port)


if __name__ == "__main__":
    main()
