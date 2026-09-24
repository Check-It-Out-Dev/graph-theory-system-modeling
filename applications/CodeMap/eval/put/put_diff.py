"""A run's change set, read from its committed artifacts: the unified diff (`diff.patch`), the files as the agent left
them (`after/`), and the base text of each file from the repository at the base commit.

    changes(run_dir, base_text)  -> {path: Change}
    Change.status        A (added), M (modified), D (deleted)
    Change.added         [text] of lines the diff adds
    Change.added_numbers {line number in the final file} of those lines
    Change.after         final text (None when deleted)
    Change.base          text at the base commit (None when added)
"""

import os
import re
import subprocess
from dataclasses import dataclass, field

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass
class Change:
    path: str
    status: str
    added: list = field(default_factory=list)
    added_numbers: set = field(default_factory=set)
    removed: list = field(default_factory=list)
    after: str = None
    base: str = None


def parse(patch_text):
    """-> {path: Change} with status and added/removed lines; texts are filled by `changes`."""
    out, cur, line_no = {}, None, 0
    for raw in patch_text.splitlines():
        if raw.startswith("diff --git "):
            m = re.match(r"diff --git a/(.+?) b/(.+)$", raw)
            path = m.group(2) if m else raw.split()[-1][2:]
            cur = out.setdefault(path, Change(path=path, status="M"))
            continue
        if cur is None:
            continue
        if raw.startswith("new file mode"):
            cur.status = "A"
        elif raw.startswith("deleted file mode"):
            cur.status = "D"
        elif raw.startswith("rename from ") or raw.startswith("rename to "):
            cur.status = "R"
        elif raw.startswith("+++ ") or raw.startswith("--- ") or raw.startswith("index ") or raw.startswith("similarity"):
            continue
        elif raw.startswith("@@"):
            m = HUNK.match(raw)
            line_no = int(m.group(1)) if m else 0
        elif raw.startswith("+"):
            cur.added.append(raw[1:])
            cur.added_numbers.add(line_no)
            line_no += 1
        elif raw.startswith("-"):
            cur.removed.append(raw[1:])
        elif raw.startswith(" "):
            line_no += 1
        elif raw.startswith("\\"):
            continue
    return out


def git_base_reader(source, base_sha):
    """-> base_text(path): the file's text at the base commit in the source clone, or None; cached."""
    cache = {}

    def read(path):
        if path not in cache:
            p = subprocess.run(["git", "show", f"{base_sha}:{path}"], cwd=source, capture_output=True)
            cache[path] = p.stdout.decode("utf-8", "replace").replace("\r\n", "\n") if p.returncode == 0 else None
        return cache[path]

    return read


def changes(run_dir, base_text):
    with open(os.path.join(run_dir, "diff.patch"), encoding="utf-8") as f:
        out = parse(f.read())
    for path, ch in out.items():
        if ch.status != "D":
            p = os.path.join(run_dir, "after", *path.split("/"))
            if os.path.exists(p):
                with open(p, encoding="utf-8", errors="replace") as f:
                    ch.after = f.read().replace("\r\n", "\n")
        if ch.status != "A":
            ch.base = base_text(path)
    return out
