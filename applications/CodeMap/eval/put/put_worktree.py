"""Disposable git worktrees of the repository under test: one per run, created at the base commit, captured as a
diff, removed. The source clone is never worked in directly (law 10).

    create(source, sha, path)   -> path          `git worktree add --detach <path> <sha>`
    capture(path)               -> unified diff  every change against the base, untracked files included
    apply(path, patch_text)                      `git apply` a patch (reference solutions, a run's diff)
    remove(source, path)                          `git worktree remove --force`, retried; then `prune`

Windows keeps handles on `target/` for a moment after Maven exits, so removal retries before it gives up.
"""

import os
import shutil
import subprocess
import time


def _git(args, cwd, check=True, input_text=None):
    if input_text is not None:
        # bytes, not text: a text-mode stdin on Windows rewrites every LF as CRLF and the patch stops matching
        p = subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True, input=input_text.encode("utf-8"))
        p.stdout = p.stdout.decode("utf-8", "replace")
        p.stderr = p.stderr.decode("utf-8", "replace")
    else:
        p = subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}: {p.stderr.strip()[:400]}")
    return p


def create(source, sha, path):
    if os.path.exists(path):
        remove(source, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _git(["worktree", "add", "--detach", path, sha], cwd=source)
    return path


def head(path):
    return _git(["rev-parse", "HEAD"], cwd=path).stdout.strip()


def capture(path):
    """-> the diff of the working tree against HEAD, new files included, binary-safe. Leaves the index as it was."""
    _git(["add", "-A"], cwd=path)
    try:
        diff = _git(["diff", "--cached", "--binary", "HEAD"], cwd=path).stdout
    finally:
        _git(["reset", "-q"], cwd=path, check=False)
    return diff


def changed_paths(path):
    """-> [(status, path)] from `git status --porcelain`, untracked expanded to files."""
    out = _git(["status", "--porcelain", "--untracked-files=all"], cwd=path).stdout
    rows = []
    for line in out.splitlines():
        if len(line) > 3:
            rows.append((line[:2].strip(), line[3:].strip().strip('"')))
    return rows


def apply(path, patch_text):
    """Apply a captured diff. It goes through the index (whose text is LF-normalised, like the diff) and is then
    checked out, because a Windows working tree holds CRLF files that a plain `git apply` cannot match."""
    if not patch_text.strip():
        return
    _git(["apply", "--cached", "--whitespace=nowarn", "-"], cwd=path, input_text=patch_text)
    _git(["checkout", "--", "."], cwd=path)


def remove(source, path, attempts=4):
    for i in range(attempts):
        p = _git(["worktree", "remove", "--force", path], cwd=source, check=False)
        if p.returncode == 0 or not os.path.exists(path):
            break
        time.sleep(1.5 * (i + 1))
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    _git(["worktree", "prune"], cwd=source, check=False)


def prune(source):
    _git(["worktree", "prune"], cwd=source, check=False)
