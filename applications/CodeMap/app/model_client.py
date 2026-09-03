# CodeMap navigator model client — owns the llama-server sidecar and exposes the ONE
# autonomous-loop implementation (imported from loop_runner — the eval and the product
# run the SAME loop; two copies would be two instruments sharing no assumptions, L11).
# Absence is a feature: no GGUF on disk -> server.py keeps the v0 zero-model protocol.

import os
import subprocess

import loop_runner  # run_loop + chat live there; single implementation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_GGUF = os.path.join(ROOT, "bin", "models", "codemap-lora-r22-q4_k_m.gguf")


class NavigatorModel:
    """Lazy llama-server sidecar + the autonomous ask-loop. One instance per process."""

    def __init__(self, gguf=None, port=7348):
        self.gguf = gguf or os.environ.get("CODEMAP_GGUF") or DEFAULT_GGUF
        self.port = port
        self.proc = None

    def available(self):
        return os.path.exists(self.gguf) and os.path.exists(loop_runner.SERVER)

    def _ensure(self):
        if self.proc and self.proc.poll() is None:
            return True
        self.proc = subprocess.Popen(
            [loop_runner.SERVER, "-m", self.gguf, "-c", "4096",
             "--port", str(self.port), "-t", str(max(2, (os.cpu_count() or 8) - 2)),
             "--no-webui", "--grammar-file", loop_runner.GRAMMAR],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return loop_runner.wait_health(self.port)

    def ask(self, engine, question):
        """Autonomous session: returns the full trajectory + terminal for the UI."""
        if not self.available():
            return None
        if not self._ensure():
            return None
        traj, terminal, text, fps, backtracks = loop_runner.run_loop(
            self.port, engine, question)
        return dict(trajectory=traj, terminal=terminal, text=text,
                    steps=len(traj), backtracks=backtracks)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
