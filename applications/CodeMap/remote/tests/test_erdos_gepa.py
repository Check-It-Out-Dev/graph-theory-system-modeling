"""GEPA over the Erdős skill without a model: the quality-first score, the guard against memorised answers,
the masked feedback, and one evaluation with a fake runner and a fake judge."""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "eval", "erdos"))

import erdos_gepa  # noqa: E402
import erdos_prompt  # noqa: E402
import run_pairs  # noqa: E402


class ScoreTests(unittest.TestCase):
    def test_quality_comes_first(self):
        same_cost = dict(erdos_weighted=500, general_weighted=1000)
        worse_cheap = erdos_gepa.score({"overall": 3}, {"overall": 4}, **same_cost)
        equal_cheap = erdos_gepa.score({"overall": 4}, {"overall": 4}, **same_cost)
        equal_dear = erdos_gepa.score({"overall": 4}, {"overall": 4}, erdos_weighted=1000, general_weighted=1000)
        self.assertEqual(worse_cheap, round(0.7 * 0.75 + 0.2 * 0.5, 4))
        self.assertEqual(equal_cheap, 0.9)
        self.assertEqual(equal_dear, 0.8)
        self.assertGreater(equal_dear, worse_cheap)                 # matching quality at no saving beats a cheap worse answer
        self.assertEqual(erdos_gepa.score({"overall": 5}, {"overall": 4}, **same_cost), 0.9)   # parity is capped
        self.assertEqual(erdos_gepa.score({}, {"overall": 4}, **same_cost), 0.0)


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = erdos_gepa.key_identifiers(run_pairs.load_problems())

    def test_the_seed_passes_and_a_memorising_candidate_is_refused(self):
        seed = erdos_prompt.skill_body()
        self.assertEqual(erdos_gepa.check(seed, self.ids), [])
        leaking = seed + "\n\nNote: during an outage the upload limiter in StorageRateLimitService fails open."
        problems = erdos_gepa.check(leaking, self.ids)
        self.assertTrue(any("StorageRateLimitService" in p for p in problems), problems)
        self.assertTrue(erdos_gepa.check(seed.replace("references/graph-map.md", "the map"), self.ids))
        self.assertTrue(erdos_gepa.check("x" * (erdos_gepa.MAX_CHARS + 1) + "references/graph-map.md", self.ids))

    def test_feedback_masks_code_names(self):
        masked = erdos_gepa.mask("B wrongly says RedisUserCache.getTokenVersion fails open; see step_up_token in StepUpAuthService.java")
        for name in ("RedisUserCache", "getTokenVersion", "step_up_token", "StepUpAuthService"):
            self.assertNotIn(name, masked)
        self.assertIn("fails open", masked)


class EvaluateTests(unittest.TestCase):
    def test_one_evaluation_with_fakes(self):
        problems = run_pairs.load_problems()[:1]
        pid = problems[0]["id"]
        seen = {}

        def runner(cmd, cwd, events_path, timeout):
            seen["cmd"] = cmd
            with open(events_path, "w", encoding="utf-8") as f:
                for t, e in [(0.0, {"type": "system", "subtype": "init"}),
                             (1.0, {"type": "stream_event", "event": {"type": "message_start", "message": {"id": "m1", "usage": {
                                 "input_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1000, "output_tokens": 1}}}}),
                             (1.1, {"type": "assistant", "message": {"id": "m1", "content": [{"type": "text", "text": "## Problem\n" + "a" * 500 + "\n=== ANSWER COMPLETE ==="}]}}),
                             (1.2, {"type": "stream_event", "event": {"type": "message_delta", "usage": {
                                 "input_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1000, "output_tokens": 99}}}),
                             (2.0, {"type": "result", "subtype": "success", "is_error": False})]:
                    f.write(json.dumps({"t": t, "event": e}) + "\n")
            return 2.0

        def judge(problem, a, b, model):
            order = erdos_gepa.erdos_judge.blind_order(problem["id"])
            letter = {order[0]: "A", order[1]: "B"}
            return ({letter["erdos"]: {"overall": 4, "correctness": 4}, letter["general"]: {"overall": 4, "correctness": 5},
                     "equivalent": True, "better": "tie", "why": "RedisUserCache claim unverified"}, {}, False)

        with tempfile.TemporaryDirectory() as d:
            ad = erdos_gepa.ErdosAdapter(problems, "2026-09-17", "C:/ws", os.path.join(R, "graph", "pack"), d, runner=runner, judge=judge,
                                         parallel=1, context_root=os.path.join(d, "ctx"))
            eb = ad.evaluate([{"id": pid}], {erdos_gepa.COMPONENT: erdos_prompt.skill_body()}, capture_traces=True)
            self.assertEqual(len(eb.scores), 1)
            self.assertGreater(eb.scores[0], 0.8)                       # parity reached, and cheaper than the reference
            ctx = seen["cmd"][seen["cmd"].index("--add-dir") + 1]
            self.assertEqual(os.listdir(ctx), ["CLAUDE.md"])
            self.assertIn("mcp__graph__*", seen["cmd"][seen["cmd"].index("--allowedTools") + 1])
            fb = eb.trajectories[0]["feedback"]
            self.assertNotIn("RedisUserCache", fb)
            rows = ad.make_reflective_dataset({}, eb, [erdos_gepa.COMPONENT])[erdos_gepa.COMPONENT]
            self.assertEqual(rows[0]["Generated Outputs"], "(answer withheld: judged against a key)")
            calls = {"judge": 0}
            ad.judge = lambda *args: calls.__setitem__("judge", calls["judge"] + 1) or judge(*args)
            again = ad.evaluate([{"id": pid}], {erdos_gepa.COMPONENT: erdos_prompt.skill_body()})   # cached run and verdict
            self.assertEqual(again.scores, eb.scores)
            self.assertEqual(calls["judge"], 0)


if __name__ == "__main__":
    unittest.main()
