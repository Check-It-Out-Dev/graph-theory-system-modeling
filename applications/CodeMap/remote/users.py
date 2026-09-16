"""The user enum: who may call, and what they may spend. Unknown -> None (the caller answers 400).

Why an enum and not free text: users are the governance dimension (credits, ratings, dashboards);
a typo must not mint a persona and label cardinality must stay bounded. Registration is a commit.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ID_RX = re.compile(r"^[a-z0-9][a-z0-9-]{1,31}$")  # Prometheus-label safe
HEADER = "X-CodeMap-User"


def load(path=None):
    path = path or os.environ.get("CODEMAP_USERS") or os.path.join(HERE, "users.json")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    users = {}
    for u in doc["users"]:
        if not ID_RX.match(u["id"]):
            raise ValueError(f"bad user id {u['id']!r}")
        users[u["id"]] = u
    return users


def resolve(users, headers, body=None):
    """body.user wins, then the header, then 'anonymous'. Returns (user_dict | None, wanted_id)."""
    wanted = None
    if body and isinstance(body.get("user"), str):
        wanted = body["user"]
    if wanted is None:
        wanted = _header(headers, HEADER)
    if not wanted:
        wanted = "anonymous"
    return users.get(wanted), wanted


def _header(headers, name):
    if headers is None:
        return None
    get = getattr(headers, "get", None)
    if get is None:
        return None
    v = get(name)
    if v is None:
        v = get(name.lower())
    return v.strip() if isinstance(v, str) else None
