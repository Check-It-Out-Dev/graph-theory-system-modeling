"""The prompt optimiser without a model: constraints refuse a candidate that drops a verb or a
placeholder (breakage 8), the adapter scores answers by the oracle with a fake CLI, a refused candidate
scores zero with the violation as feedback, the reflective dataset has the three columns GEPA reads,
the dataset builder keeps only what the oracle can check, and the promotion gate says no unless every
condition holds."""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "eval", "optimize"))

import adapter as adapter_mod  # noqa: E402
import constraints  # noqa: E402
import promote  # noqa: E402

PACK = os.path.join(R, "graph", "pack")
HAS_PACK = os.path.exists(os.path.join(PACK, "entities.csv"))
TEMPLATE = open(os.path.join(R, "prompts", "navigator", "template.md"), encoding="utf-8").read()


def _runner_answer(text):
    class P:
        def __init__(self):
            self.stdout = json.dumps({"type": "result", "is_error": False, "result": text, "session_id": "s", "num_turns": 2,
                                      "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 5000, "cache_creation_input_tokens": 0}})
            self.stderr, self.returncode = "", 0
    return lambda cmd, env, cwd, timeout: P()


class ConstraintTests(unittest.TestCase):
    def test_the_seed_template_passes(self):
        self.assertEqual(constraints.check(TEMPLATE), [])

    def test_breakage_8_a_verb_removed(self):
        broken = TEMPLATE.replace("seam(", "sea_m(")
        probs = constraints.check(broken)
        self.assertTrue(any("verb seam()" in p for p in probs), probs)

    def test_placeholders_and_contract(self):
        probs = constraints.check(TEMPLATE.replace("{{L2_NAVIGATORS}}", ""))
        self.assertTrue(any("{{L2_NAVIGATORS}} appears 0" in p for p in probs))
        probs = constraints.check(TEMPLATE.replace("```json", "```"))
        self.assertTrue(any("answer contract" in p for p in probs))
        self.assertTrue(any("words <" in p for p in constraints.check("{{L1_INDEX}} {{CAVEATS}} {{L2_NAVIGATORS}} {{CURATION_NOTES}}")))


@unittest.skipUnless(HAS_PACK, "pack absent")
class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.notes = os.path.join(R, "prompts", "navigator", "curation_notes.md")
        self.row = {"id": "BE01", "archetype": "locate", "gold_status": "EXECUTED", "q": "What does PaymentsDisabledBootGuard do?",
                    "gold_answer": "PaymentsDisabledBootGuard.java stops the app", "gold_result_excerpt": [["PaymentsDisabledBootGuard.java", "Rule", "11", "x"]]}
        self.inst = {"id": "BE01", "kind": "bank", "q": self.row["q"], "row": self.row}
        self.probe = {"id": "OD01", "kind": "probe", "q": "Where is the quantum flux capacitor?", "row": {"id": "OD01", "kind": "off", "expect": "abstain"}}

    def test_scores_by_the_oracle(self):
        good = "It gates boot.\n```json\n{\"terminal\": \"answer\", \"pointers\": [\"PaymentsDisabledBootGuard.java\"]}\n```"
        ad = adapter_mod.NavigatorAdapter(PACK, self.notes, runner=_runner_answer(good))
        eb = ad.evaluate([self.inst, self.probe], {adapter_mod.COMPONENT: TEMPLATE}, capture_traces=True)
        self.assertEqual(eb.scores, [1.0, 0.0])  # the probe expected an abstention
        self.assertIn("correct", eb.trajectories[0]["feedback"])
        self.assertIn("expected 'abstain'", eb.trajectories[1]["feedback"])
        self.assertEqual(ad.calls, 2)
        self.assertEqual(ad.usage["cache_read_input_tokens"], 10000)
        ds = ad.make_reflective_dataset({adapter_mod.COMPONENT: TEMPLATE}, eb, [adapter_mod.COMPONENT])
        rows = ds[adapter_mod.COMPONENT]
        self.assertEqual(sorted(rows[0].keys()), ["Feedback", "Generated Outputs", "Inputs"])
        miss = "Nope.\n```json\n{\"terminal\": \"answer\", \"pointers\": [\"Other.java\"]}\n```"
        eb2 = adapter_mod.NavigatorAdapter(PACK, self.notes, runner=_runner_answer(miss)).evaluate([self.inst], {adapter_mod.COMPONENT: TEMPLATE}, True)
        self.assertEqual(eb2.scores, [0.0])
        self.assertIn("missed", eb2.trajectories[0]["feedback"])

    def test_a_candidate_that_breaks_the_contract_is_refused_without_a_call(self):
        ad = adapter_mod.NavigatorAdapter(PACK, self.notes, runner=_runner_answer("x"))
        eb = ad.evaluate([self.inst], {adapter_mod.COMPONENT: TEMPLATE.replace("{{L1_INDEX}}", "")}, capture_traces=True)
        self.assertEqual((eb.scores, ad.calls), ([0.0], 0))
        self.assertIn("refused", eb.trajectories[0]["feedback"])

    def test_dataset_keeps_only_checkable_rows(self):
        train, val = adapter_mod.dataset(PACK, os.path.join(R, "eval", "q", "probes_offdist.jsonl"), 6, 6, seed=1)
        self.assertEqual((len(train), len(val)), (6, 6))
        self.assertFalse(set(i["id"] for i in train) & set(i["id"] for i in val))
        for i in train + val:
            self.assertIn(i["kind"], ("bank", "probe"))
            if i["kind"] == "bank":
                self.assertIn(i["row"]["archetype"], ("locate", "impact", "flow", "boundary", "onboarding", "onboarding_path", "cohort", "health"))

    def test_reflection_lm_is_one_toolless_call(self):
        seen = {}

        def runner(cmd, env, cwd, timeout):
            seen["cmd"] = cmd
            return _runner_answer("<new prompt>")(cmd, env, cwd, timeout)
        teacher = adapter_mod.reflection_lm(runner=runner)
        self.assertEqual(teacher("improve this"), "<new prompt>")
        self.assertIn("--tools", seen["cmd"])
        self.assertEqual(teacher.usage["calls"], 1)


class PromoteTests(unittest.TestCase):
    def test_gate(self):
        run = {"mode": "gepa", "win": True, "seed_val_score": 0.5, "best_val_score": 0.6}
        cand = TEMPLATE.replace("Do not restate the index", "Never restate the index")
        ok, reasons = promote.decide(run, cand, TEMPLATE)
        self.assertTrue(ok, reasons)
        self.assertFalse(promote.decide({"mode": "gepa", "win": True, "seed_val_score": 0.5, "best_val_score": 0.52}, cand, TEMPLATE)[0])
        self.assertFalse(promote.decide(run, TEMPLATE, TEMPLATE)[0])  # identical
        self.assertFalse(promote.decide(run, cand.replace("seam(", "x("), TEMPLATE)[0])  # constraint
        self.assertFalse(promote.decide({"mode": "dry-run", "win": False}, cand, TEMPLATE)[0])


if __name__ == "__main__":
    unittest.main()
