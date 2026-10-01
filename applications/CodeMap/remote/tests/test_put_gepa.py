"""GEPA over the conventions manual without a model: the plateau stopper, the guard against memorised tasks and dropped
rules, the masked feedback, and one evaluation with a fake runner and a fake judge over a committed fixture."""

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "eval", "put"))
sys.path.insert(0, HERE)

import put_contract  # noqa: E402
import put_gepa  # noqa: E402
import put_paths  # noqa: E402
import put_prompt  # noqa: E402
import put_stop  # noqa: E402
from test_put_checks import FIX, FixtureRepo  # noqa: E402

INSTANCE = "backend-conventions"


class StopperTests(unittest.TestCase):
    def test_k_iterations_without_a_gain_above_delta_stop_the_run(self):
        s = put_stop.PlateauStopper(k=3, delta=0.02)
        self.assertFalse(s.observe([0.60], 0))                 # the seed sets the anchor
        self.assertFalse(s.observe([0.60, 0.70], 1))           # a real gain moves it
        self.assertFalse(s.observe([0.60, 0.70, 0.71], 2))     # +0.01 is inside the noise: the streak runs
        self.assertFalse(s.observe([0.60, 0.70, 0.71], 3))
        self.assertTrue(s.observe([0.60, 0.70, 0.71], 4))      # three iterations after the anchor
        self.assertEqual(s.streak(), 3)

    def test_the_streak_survives_a_resume(self):
        path = os.path.join(tempfile.mkdtemp(), "plateau.json")
        s = put_stop.PlateauStopper(3, 0.02, path)
        s.observe([0.5], 0)
        s.observe([0.5], 2)
        again = put_stop.PlateauStopper(3, 0.02, path)
        self.assertTrue(again.observe([0.5], 3))

    def test_the_gepa_state_is_read_through_its_properties(self):
        class State:
            program_full_scores_val_set = [0.4, 0.45]
            i = 5
        s = put_stop.PlateauStopper(3, 0.1)
        self.assertFalse(s(State()))


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.contract = put_contract.load(INSTANCE)
        self.seed = put_prompt.load_body(INSTANCE)
        self.ids = put_gepa.task_identifiers(INSTANCE, put_contract.tasks(INSTANCE))

    def test_the_seed_passes_the_guard(self):
        self.assertEqual(put_gepa.guard(self.seed, self.contract, self.seed, self.ids), [])

    def test_a_manual_that_memorises_a_task_is_refused(self):
        self.assertIn("StaleTicketCloseCronJob", self.ids)
        leaked = self.seed.replace("</orientation>", "Name the job StaleTicketCloseCronJob.\n</orientation>")
        self.assertTrue(any("names code from the tasks" in p for p in put_gepa.guard(leaked, self.contract, self.seed, self.ids)))

    def test_a_manual_that_drops_a_rule_or_an_include_is_refused(self):
        dropped = self.seed.replace('<rule id="scheduler_lock">', "<rule>")
        self.assertIn("rule ids dropped: scheduler_lock", put_gepa.guard(dropped, self.contract, self.seed, self.ids))
        no_inc = self.seed.replace('<include file="references/tools.md"/>', "")
        self.assertIn("the manual no longer includes references/tools.md", put_gepa.guard(no_inc, self.contract, self.seed, self.ids))

    def test_a_manual_over_the_limit_is_refused(self):
        big = self.seed + "x" * 30000
        self.assertTrue(any("longer than" in p for p in put_gepa.guard(big, self.contract, self.seed, self.ids)))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.contract = put_contract.load(INSTANCE)
        self.seed = put_prompt.load_body(INSTANCE)
        self.label = "test-put-gepa-" + next(tempfile._get_candidate_names())
        self.tasks = [t for t in put_contract.tasks(INSTANCE) if t["split"] == "train"]

        def fake_execute(task, body, rd, source, contract, instance, label=None):
            shutil.copytree(os.path.join(FIX, f"{task['id']}--reference"), rd, dirs_exist_ok=True)
            with open(os.path.join(rd, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"task": task["id"], "marker": True, "session": {"budget_exhausted": False}}, f)

        verdict = {"correctness": 5, "convention_fit": 4, "design_fit": 4, "test_quality": 3, "graph_use": 5,
                   "reasons": {"design_fit": "StaleTicketCloseCronJob duplicates SupportTicketService logic"}}
        self.ad = put_gepa.PutAdapter(INSTANCE, self.contract, self.tasks, FIX, self.label, self.seed, parallel=2,
                                      execute=fake_execute, judge=lambda *a, **k: {"verdict": verdict}, repo=FixtureRepo())

    def tearDown(self):
        shutil.rmtree(os.path.join(put_paths.RUNS, self.label), ignore_errors=True)

    def test_an_evaluation_scores_each_task_once_and_masks_the_feedback(self):
        batch = [{"id": "stale-ticket-close"}, {"id": "faq-reorder"}, {"id": "stale-ticket-close"}]
        out = self.ad.evaluate(batch, {put_gepa.COMPONENT: self.seed}, capture_traces=True)
        self.assertEqual(len(out.scores), 3)
        self.assertEqual(self.ad.calls, 2)                                   # a repeated task runs once
        self.assertGreater(out.scores[0], 0.8)
        fb = out.trajectories[0]["feedback"]
        self.assertIn("rule scheduler_lock: followed", fb)
        self.assertNotIn("StaleTicketCloseCronJob", fb)                      # code names masked
        rows = self.ad.make_reflective_dataset({}, out, [put_gepa.COMPONENT])[put_gepa.COMPONENT]
        self.assertEqual(rows[0]["Generated Outputs"], "(diff withheld: graded by the checks and the reviewer)")

    def test_a_refused_candidate_scores_zero_without_running(self):
        leaked = self.seed.replace("</orientation>", "Use StaleTicketCloseCronJob.\n</orientation>")
        out = self.ad.evaluate([{"id": "stale-ticket-close"}], {put_gepa.COMPONENT: leaked}, capture_traces=True)
        self.assertEqual(out.scores, [0.0])
        self.assertEqual(self.ad.calls, 0)
        self.assertIn("refused before running", out.trajectories[0]["feedback"])


if __name__ == "__main__":
    unittest.main()
