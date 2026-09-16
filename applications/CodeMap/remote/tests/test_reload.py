"""Reload and the pack poller with a fake fetch: the engine is swapped, versions move, a reload
event is written, the admin route needs both tokens, the poller reloads only on exit code 3."""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)

from remote import reload as reload_mod  # noqa: E402

PACK = os.path.join(R, "graph", "pack")
HAS_PACK = os.path.exists(os.path.join(PACK, "entities.csv")) and os.path.exists(os.path.join(PACK, "codemap.lbdb"))


@unittest.skipUnless(HAS_PACK, "graph pack absent")
class ReloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from remote import server
        cls.server = server
        os.environ["CODEMAP_TELEMETRY_DIR"] = tempfile.mkdtemp()
        cls.sink = []
        cls.app = server.App(token="t", admin_token="adm", sink=cls.sink, navigator=False)

    def test_active_prompt_is_built_from_the_pack(self):
        self.assertTrue(self.app.prompt_path.endswith("navigator-active.md"))
        text = open(self.app.prompt_path, encoding="utf-8").read()
        self.assertIn("L1 SUBSYSTEM INDEX", text)
        self.assertTrue(self.app.prompt_version.startswith("nav@"))

    def test_reload_swaps_engine_and_writes_an_event(self):
        old_engine = self.app.engine
        n = len(self.sink)
        ok, detail = self.app.reload(runner=lambda cmd: (True, "fetched (fake)"))
        self.assertTrue(ok, detail)
        self.assertIsNot(self.app.engine, old_engine)
        self.assertEqual(self.sink[n]["event_type"], "reload")
        self.assertGreater(len(self.app.engine.ents), 1400)  # whatever pack the Release holds
        self.assertTrue(self.app.reloads[-1]["ok"])

    def test_failed_fetch_keeps_the_old_engine(self):
        old_engine = self.app.engine
        ok, detail = self.app.reload(runner=lambda cmd: (False, "sha256 mismatch"))
        self.assertFalse(ok)
        self.assertIs(self.app.engine, old_engine)
        self.assertIn("mismatch", detail)

    def test_admin_route_needs_both_tokens(self):
        self.assertEqual(self.server.handle_reload(self.app, {"Authorization": "Bearer t"})[0], 401)
        self.assertEqual(self.server.handle_reload(self.app, {"X-CodeMap-Admin": "adm"})[0], 401)
        self.assertEqual(self.server.handle_reload(self.app, {"Authorization": "Bearer wrong", "X-CodeMap-Admin": "adm"})[0], 401)

    def test_poller_reloads_only_on_a_newer_release(self):
        calls = []

        def runner(cmd):
            calls.append(cmd)
            return 0 if "--check" in cmd else (True, "ok")
        p = reload_mod.Poller(self.app, interval_s=0, runner=runner)
        self.assertFalse(p.tick())
        self.assertEqual(p.last["result"], 0)
        p.runner = lambda cmd: 3 if "--check" in cmd else (True, "ok")
        self.assertTrue(p.tick())
        self.assertEqual(p.last["reloaded"], 1)
        self.assertFalse(p.start())  # interval 0 = disabled


class HousekeepingTests(unittest.TestCase):
    def test_prune_old_sessions_only(self):
        import time
        from remote import housekeeping
        d = tempfile.mkdtemp()
        old = os.path.join(d, "old.jsonl")
        new = os.path.join(d, "new.jsonl")
        open(old, "w").write("x")
        open(new, "w").write("y")
        os.utime(old, (time.time() - 10 * 86400, time.time() - 10 * 86400))
        self.assertEqual(housekeeping.prune_sessions(d, keep_days=7), (1, 1))
        self.assertTrue(os.path.exists(new) and not os.path.exists(old))
        self.assertEqual(housekeeping.prune_sessions(os.path.join(d, "nope")), (0, 0))
        self.assertTrue(housekeeping.sessions_dir("/home/codemap/cli", home="/h").endswith(os.path.join(".claude", "projects", "-home-codemap-cli")))


if __name__ == "__main__":
    unittest.main()
