"""Pack reload: fetch the latest Release, swap the engine, rebuild the active prompt and the search
index, without a restart. Triggered by POST /admin/reload (the deploy) or by the poll thread that
asks GitHub every CODEMAP_PACK_POLL_S seconds whether a newer pack-* release exists.

The active prompt is BUILT here (template + pack data + curation notes) into the telemetry dir and
identified by its own sha16, so a new pack changes prompt_version too: drift is attributable to the
(prompt, pack) pair the event carries, not to a mystery.
"""

import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
FETCH = os.path.join(R, "tools", "pack", "fetch_pack.py")


def fetch_latest(dest, runner=None):
    """-> (ok, output). Runs fetch_pack.py --latest in a subprocess (it is stdlib, but isolation keeps
    a bad download from touching the live engine until the files are verified and extracted)."""
    cmd = [sys.executable, FETCH, "--latest", "--dest", dest]
    if runner is not None:
        return runner(cmd)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # NOSONAR - argv list, our own script; see sonar-project.properties
    return p.returncode == 0, (p.stdout + p.stderr)[-800:]


def check_latest(dest, runner=None):
    """-> 0 same, 3 newer available, other = error."""
    cmd = [sys.executable, FETCH, "--check", "--dest", dest]
    if runner is not None:
        return runner(cmd)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # NOSONAR - argv list, our own script; see sonar-project.properties
    return p.returncode


def build_active_prompt(app, out_path):
    sys.path.insert(0, os.path.join(R, "tools", "prompt"))
    import build_navigator
    template = os.path.join(R, "prompts", "navigator", "template.md")
    notes = os.path.join(R, "prompts", "navigator", "curation_notes.md")
    text, _ = build_navigator.build(template, app.pack_dir, notes)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return out_path


class Poller:
    def __init__(self, app, interval_s=None, runner=None):
        self.app = app
        self.interval = int(os.environ.get("CODEMAP_PACK_POLL_S", "600") if interval_s is None else interval_s)
        self.runner = runner
        self._stop = threading.Event()
        self.last = {"checked": None, "result": None, "reloaded": 0}

    def tick(self):
        rc = check_latest(self.app.pack_dir, self.runner)
        self.last["checked"] = time.time()
        self.last["result"] = rc
        if rc == 3:
            ok, out = self.app.reload(runner=self.runner)
            if ok:
                self.last["reloaded"] += 1
            return ok
        return False

    def start(self):
        if self.interval <= 0:
            return False
        threading.Thread(target=self._loop, name="codemap-pack-poll", daemon=True).start()
        return True

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception:  # never let the poller die
                pass

    def stop(self):
        self._stop.set()
