"""Housekeeping the server does for itself: prune the navigator's Claude Code session transcripts.

Every `claude -p` turn persists a session file under ~/.claude/projects/<cwd slug>/ so that
`--resume` works for follow-ups; conversations are short-lived, so anything older than KEEP_DAYS is
dead weight on the volume. Runs at boot and on every pack-poll tick. Never touches anything outside
the CLI's own project directory for the bare working directory.
"""

import os
import time

KEEP_DAYS = int(os.environ.get("CODEMAP_SESSION_KEEP_DAYS", "7"))


def sessions_dir(cwd=None, home=None):
    """Claude Code keeps sessions under ~/.claude/projects/<slug>, slug = the cwd with separators as dashes."""
    home = home or os.path.expanduser("~")
    cwd = cwd or os.environ.get("CODEMAP_CLI_CWD") or ""
    if not cwd:
        return None
    slug = cwd.replace("\\", "/").replace("/", "-").replace(":", "-")
    return os.path.join(home, ".claude", "projects", slug)


def prune_sessions(directory=None, keep_days=None, now=None):
    """Delete session files older than keep_days; -> (deleted, kept). Silent on a missing dir."""
    directory = directory or sessions_dir()
    keep_days = KEEP_DAYS if keep_days is None else keep_days
    if not directory or not os.path.isdir(directory):
        return 0, 0
    now = now or time.time()
    cutoff = now - keep_days * 86400
    deleted = kept = 0
    for root, _, files in os.walk(directory):
        for name in files:
            p = os.path.join(root, name)
            try:
                if os.path.getmtime(p) < cutoff:
                    os.remove(p)
                    deleted += 1
                else:
                    kept += 1
            except OSError:
                kept += 1
    return deleted, kept
