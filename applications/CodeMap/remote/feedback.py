"""Feedback and misses: the two signals users give back, validated strictly, recorded as events.

A rating references an answer the server actually gave (its request_id is in the seen ring,
rebuilt from ask events at boot); exactly one of rating/vote; tags from the closed list; a
repeat by the same user supersedes in the metrics view. A miss is a file the graph did not
know: it goes to the saturation backlog the delta pipeline indexes next.
"""

import json
import os
import re
import threading
from collections import OrderedDict

from . import mcp, telemetry, tools as tools_mod

MAX_COMMENT = 300
MAX_WHY = 200
SEEN_CAP = 20000
_PATH_RX = re.compile(r"^[A-Za-z0-9_./\-@+]{1,300}$")


class Seen:
    """request_id -> {tier, user, ts} for answers given; bounded, rebuilt from events."""

    def __init__(self, cap=SEEN_CAP):
        self.cap = cap
        self._d = OrderedDict()
        self._lock = threading.Lock()

    def observe(self, ev):
        if ev.get("event_type") == "ask" and ev.get("request_id"):
            with self._lock:
                self._d[ev["request_id"]] = {"tier": ev.get("tier"), "user": ev.get("user"), "ts": ev.get("ts"),
                                             "terminal": ev.get("terminal")}
                self._d.move_to_end(ev["request_id"])
                while len(self._d) > self.cap:
                    self._d.popitem(last=False)

    def get(self, request_id):
        with self._lock:
            return self._d.get(request_id)

    def __len__(self):
        return len(self._d)


def validate_feedback(args, seen):
    """-> (error_message | None, cleaned)"""
    rid = args.get("request_id")
    if not isinstance(rid, str) or not rid:
        return "request_id required", None
    ref = seen.get(rid)
    if ref is None:
        return "unknown request_id (not an answer this server gave, or too old)", None
    rating, vote = args.get("rating"), args.get("vote")
    if (rating is None) == (vote is None):
        return "exactly one of rating (1-5) or vote (up|down)", None
    if rating is not None and (not isinstance(rating, int) or isinstance(rating, bool) or not 1 <= rating <= 5):
        return "rating must be an integer 1-5", None
    if vote is not None and vote not in ("up", "down"):
        return "vote must be up or down", None
    tags = args.get("tags") or []
    if not isinstance(tags, list) or any(t not in mcp.FEEDBACK_TAGS for t in tags):
        return f"tags must be a subset of {mcp.FEEDBACK_TAGS}", None
    comment = args.get("comment")
    if comment is not None and (not isinstance(comment, str) or len(comment) > MAX_COMMENT):
        return f"comment must be a string of at most {MAX_COMMENT} characters", None
    verified = args.get("verified")
    if verified is not None and not isinstance(verified, bool):
        return "verified must be a boolean", None
    if rating is not None and rating > 3 and not verified:
        return "a rating above 3 requires verified=true (open at least one pointer in your checkout first)", None
    return None, {"request_id": rid, "rating": rating, "vote": vote, "tags": sorted(set(tags)),
                  "comment": comment, "verified": bool(verified), "ref": ref}


def feedback(app, ident, args):
    err, fb = validate_feedback(args, app.seen)
    if err:
        return json.dumps({"error": err}), True
    ref = fb.pop("ref")
    ev = {"schema": 1, "event_type": "feedback", "ts": telemetry.now_iso(), "request_id": fb["request_id"],
          "session_id": ident.session_id, "user": ident.user["id"], "user_kind": ident.user["kind"],
          "tier": ref.get("tier") or "none", "rating": fb["rating"], "vote": fb["vote"], "tags": fb["tags"],
          "comment": fb["comment"], "verified": fb["verified"], "credits": 0.0, "duration_ms": ident.ms(),
          "pack_version": app.pack_version, "prompt_version": app.prompt_version, "tool": "codemap_feedback"}
    app.emit(ev)
    return json.dumps({"ok": True, "request_id": fb["request_id"], "recorded": {k: fb[k] for k in ("rating", "vote", "tags", "verified")},
                       "thanks": "recorded; ratings feed the judge calibration and the prompt loop"}), False


def miss(app, ident, args):
    path = (args.get("path") or "").strip().replace("\\", "/")
    why = args.get("why")
    if not _PATH_RX.match(path) or ".." in path:
        return json.dumps({"error": "path must be a repository-relative path (repo/dir/file), no .."}), True
    if why is not None and (not isinstance(why, str) or len(why) > MAX_WHY):
        return json.dumps({"error": f"why must be at most {MAX_WHY} characters"}), True
    row = {"ts": telemetry.now_iso(), "user": ident.user["id"], "path": path, "why": why}
    try:
        os.makedirs(os.path.dirname(app.backlog_path), exist_ok=True)
        with app.backlog_lock:
            with open(app.backlog_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass
    app.emit(tools_mod.base_event(app, ident, "miss", "none", path=path, why=why, credits=0.0, tool="codemap_miss"))
    return json.dumps({"ok": True, "path": path, "note": "queued for the next reindex (saturation backlog)"}), False
