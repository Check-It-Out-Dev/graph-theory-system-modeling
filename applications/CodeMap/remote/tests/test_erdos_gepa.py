"""GEPA over Erdős's manual without a model: the score over the judge's criteria and adherence, the guards against
memorised answers, the masked feedback, one evaluation with a fake runner and a fake judge, the seed-run import and
the judge-noise measurement."""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "eval", "erdos"))
sys.path.insert(0, HERE)

import erdos_gepa  # noqa: E402
import erdos_prompt  # noqa: E402
import run_pairs  # noqa: E402
from test_erdos_adherence import _events  # noqa: E402

PERFECT = {"correctness": 5, "completeness": 5, "architecture_fit": 5, "graph_use": 5}
WORST = {"correctness": 1, "completeness": 1, "architecture_fit": 1, "graph_use": 1}


class ScoreTests(unittest.TestCase):
    def test_the_score_follows_the_criteria(self):
        self.assertAlmostEqual(sum(erdos_gepa.WEIGHTS.values()), 1.0)
        self.assertEqual(erdos_gepa.score(PERFECT, 1.0), 1.0)
        self.assertEqual(erdos_gepa.score(WORST, 0.0), 0.0)
        self.assertEqual(erdos_gepa.score({}, 1.0), 0.0)                              # an ungraded answer scores nothing
        graph_better = erdos_gepa.score(dict(PERFECT, graph_use=5, correctness=3), 0.5)
        graph_worse = erdos_gepa.score(dict(PERFECT, graph_use=1, correctness=3), 0.5)
        self.assertAlmostEqual(graph_better - graph_worse, erdos_gepa.WEIGHTS["graph_use"])


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = erdos_gepa.key_identifiers(run_pairs.load_problems())

    def test_the_seed_passes_and_a_memorising_candidate_is_refused(self):
        seed = erdos_prompt.skill_body()
        self.assertEqual(erdos_gepa.check(seed, self.ids), [])
        leaking = seed + "\n\nNote: during an outage the upload limiter in StorageRateLimitService fails open."
        self.assertTrue(any("StorageRateLimitService" in p for p in erdos_gepa.check(leaking, self.ids)))
        dropped = seed.replace('<include file="references/topology.md"/>', "")
        self.assertIn("the skill no longer includes references/topology.md", erdos_gepa.check(dropped, self.ids))
        self.assertTrue(erdos_gepa.check(seed + "x" * erdos_gepa.MAX_CHARS, self.ids))

    def test_generic_file_suffixes_are_not_identifiers(self):
        for generic in ("spec.ts", "component.ts", "service.ts", "component.html"):
            self.assertNotIn(generic, self.ids)
        self.assertIn("StorageRateLimitService", self.ids)

    def test_the_reflection_prompt_asks_for_behaviour_within_the_structure(self):
        template = erdos_gepa.REFLECTION_TEMPLATE
        for placeholder in ("<curr_param>", "<side_info>"):
            self.assertEqual(template.count(placeholder), 1)
        for rule in ("Change behaviour, not knowledge", "every <include file=", "28,000 characters", "no separate verification"):
            self.assertIn(rule, template)
        self.assertLess(erdos_gepa.TARGET_CHARS, erdos_gepa.MAX_CHARS)
        self.assertLess(len(erdos_prompt.skill_body()), erdos_gepa.TARGET_CHARS)

    def test_feedback_masks_code_names(self):
        masked = erdos_gepa.mask("B wrongly says RedisUserCache.getTokenVersion fails open; see step_up_token in StepUpAuthService.java")
        for name in ("RedisUserCache", "getTokenVersion", "step_up_token", "StepUpAuthService"):
            self.assertNotIn(name, masked)
        self.assertIn("fails open", masked)


def _write_events(path):
    with open(path, "w", encoding="utf-8") as f:
        for t, e in _events():
            f.write(json.dumps({"t": t, "event": e}) + "\n")


class EvaluateTests(unittest.TestCase):
    def test_one_evaluation_with_fakes(self):
        problems = run_pairs.load_problems()[:1]
        pid = problems[0]["id"]
        seen = {"runs": 0, "judge": 0}

        def runner(cmd, cwd, events_path, timeout):
            seen["runs"] += 1
            seen["cmd"] = cmd
            _write_events(events_path)
            return 2.0

        def judge(problem, answer, trace, model):
            seen["judge"] += 1
            seen["trace"] = trace
            verdict = dict(PERFECT, graph_use=3, must_find_hits=1, key_facts_supported=2, gaps_found=1, red_flags_made=0,
                           patterns_followed=2, reasons={"correctness": "RedisUserCache claim holds", "completeness": "misses gaps",
                                                         "architecture_fit": "reuses the guard", "graph_use": "few queries"})
            return verdict, {}, False

        with tempfile.TemporaryDirectory() as d:
            ad = erdos_gepa.ErdosAdapter(problems, "C:/ws", os.path.join(R, "graph", "pack"), d, runner=runner, judge=judge,
                                         parallel=1, context_root=os.path.join(d, "ctx"))
            seed = {erdos_gepa.COMPONENT: erdos_prompt.skill_body()}
            eb = ad.evaluate([{"id": pid}], seed, capture_traces=True)
            out = eb.outputs[0]
            self.assertEqual(eb.scores[0], erdos_gepa.score(out["verdict"], out["adherence"]["score"]))
            ctx = seen["cmd"][seen["cmd"].index("--add-dir") + 1]
            self.assertEqual(os.listdir(ctx), ["CLAUDE.md"])
            self.assertIn("mcp__graph__*", seen["cmd"][seen["cmd"].index("--allowedTools") + 1])
            self.assertIn("Graph queries: 2 in all", seen["trace"])
            fb = eb.trajectories[0]["feedback"]
            self.assertNotIn("RedisUserCache", fb)
            self.assertIn("graph_use 3/5", fb)
            self.assertIn("adherence to the manual", fb)
            rows = ad.make_reflective_dataset({}, eb, [erdos_gepa.COMPONENT])[erdos_gepa.COMPONENT]
            self.assertEqual(rows[0]["Generated Outputs"], "(answer withheld: graded against a key)")
            again = ad.evaluate([{"id": pid}], seed)                                   # cached run and verdict
            self.assertEqual(again.scores, eb.scores)
            self.assertEqual((seen["runs"], seen["judge"]), (1, 1))
            noise = erdos_gepa.judge_noise(ad, seed[erdos_gepa.COMPONENT])             # the repeat grading is a new call
            self.assertEqual((seen["judge"], noise["problems"], noise["graph_use"]), (2, 1, 0.0))


class SeedRunTests(unittest.TestCase):
    def test_seed_runs_are_imported_only_for_the_same_manual(self):
        seed = erdos_prompt.skill_body()
        with tempfile.TemporaryDirectory() as d:
            runs, run_dir = os.path.join(d, "runs"), os.path.join(d, "gepa-run")
            os.makedirs(os.path.join(runs, "old"))
            _write_events(os.path.join(runs, "old", "p.erdos.events.jsonl"))
            with open(os.path.join(runs, "old.json"), "w", encoding="utf-8") as f:
                json.dump({"meta": {"prompt_version": "erdos@0000000000000000"}}, f)
            self.assertIn("ran erdos@0000000000000000", erdos_gepa.import_seed_runs("old", seed, run_dir, ["p"], runs_dir=runs))
            with open(os.path.join(runs, "old.json"), "w", encoding="utf-8") as f:
                json.dump({"meta": {"prompt_version": erdos_prompt.version(erdos_prompt.assemble(seed))}}, f)
            self.assertEqual(erdos_gepa.import_seed_runs("old", seed, run_dir, ["p"], runs_dir=runs), ["p"])
            self.assertTrue(os.path.exists(os.path.join(run_dir, erdos_gepa.sha16(seed), "p.erdos.events.jsonl")))


class ReportTests(unittest.TestCase):
    def test_the_report_reads_the_log_and_the_best_candidates_change(self):
        import erdos_gepa_report
        seed = "<m>\n<core_rules>\none\n</core_rules>\n<method>\nread\n</method>\n</m>\n"
        best = "<m>\n<core_rules>\none\n</core_rules>\n<method>\nquery first, then read\n</method>\n</m>\n"
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "g")
            for sha, body in (("s0", seed), ("c1", best)):
                os.makedirs(os.path.join(run, sha))
                with open(os.path.join(run, sha, "skill_body.md"), "w", encoding="utf-8") as f:
                    f.write(body)
            checks = {"graph_pass": 0.8, "key_files_read": 0.5, "facts_grounded": 1.0, "one_pass": 1.0, "contract": 1.0}
            with open(os.path.join(run, "gepa.log.jsonl"), "w", encoding="utf-8") as f:
                for sha, value in (("s0", 0.6), ("c1", 0.8)):
                    f.write(json.dumps({"event": "eval", "candidate": sha, "problem": "p", "score": value, "adherence": 0.9, "checks": checks,
                                        "scores": {"correctness": 4, "completeness": 3, "architecture_fit": 4, "graph_use": 5}}) + "\n")
                f.write(json.dumps({"event": "refused", "candidate": "x", "problems": ["the skill names code"]}) + "\n")
            with open(os.path.join(d, "g.json"), "w", encoding="utf-8") as f:
                json.dump({"mission": "m", "weights": erdos_gepa.WEIGHTS, "seed_val_score": 0.6, "best_idx": 1, "best_val_score": 0.8,
                           "candidates": [{"sha": "s0", "val_score": 0.6}, {"sha": "c1", "val_score": 0.8}], "improved": True,
                           "judge_noise": {"correctness": 0.4, "score": 0.05, "problems": 5}}, f)
            text = erdos_gepa_report.build("g", runs_dir=d)
            self.assertIn("| 1 | `c1` | 0.8 | 0.8 | 4.0 | 3.0 | 4.0 | 5.0 | 0.9 |", text)
            self.assertIn("| method | 3 | 3 | yes |", text)
            self.assertIn("| core_rules | 3 | 3 |  |", text)
            self.assertIn("+query first, then read", text)
            self.assertIn("Judge noise on the seed", text)
            self.assertIn("## Refused candidates (1)", text)


if __name__ == "__main__":
    unittest.main()
