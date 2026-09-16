"""Pointers, not content: the graph pins WHERE; the reader opens the file in their own checkout.

Entity paths in the pack are absolute paths of the authoring box; a pointer rewrites them into
(repo, relative path) so a persona on any machine can open the file.
"""

import re

_REPOS = (
    (re.compile(r"checkItOut-be2/"), "backend"),
    (re.compile(r"checkitout-backend/"), "backend"),
    (re.compile(r"checkItOut-fe-greenfield/"), "frontend"),
    (re.compile(r"checkitout-frontend/"), "frontend"),
    (re.compile(r"checkItOut-fe/"), "frontend-legacy"),
)


def split_path(abs_path):
    """'C:/.../checkItOut-be2/src/x.java' -> ('backend', 'src/x.java'); unknown -> ('?', basename)."""
    p = (abs_path or "").replace("\\", "/")
    for rx, repo in _REPOS:
        m = rx.search(p)
        if m:
            return repo, p[m.end():]
    return "?", p.rsplit("/", 1)[-1]


def to_pointer(entity, clue=None):
    repo, rel = split_path(entity.get("file_path", ""))
    ptr = {
        "name": entity.get("name"),
        "repo": repo,
        "path": rel,
        "type": entity.get("entity_type") or entity.get("type"),
        "subsystem": _int(entity.get("curated") or entity.get("subsystem")),
        "lines": _int(entity.get("line_count")),
        "entry_point": str(entity.get("entry_point", "")).lower() == "true",
    }
    if clue:
        ptr["clue"] = clue
    return ptr


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None
