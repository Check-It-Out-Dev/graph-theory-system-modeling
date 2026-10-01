"""The prompt-under-test instance backend-conventions is consistent before any run: every rule the prompt states has
a check, the weights sum to one, every tag a rule selects has a training task and a held-out task, and the seed
prompt stays within its size and keeps its includes. No model, no Maven."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "eval", "put"))

import put_contract  # noqa: E402

INSTANCE = "backend-conventions"


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = put_contract.load(INSTANCE)
        self.prompt = put_contract.prompt_text(INSTANCE, "v1")
        self.tasks = put_contract.tasks(INSTANCE)

    def test_the_instance_is_consistent(self):
        self.assertEqual(put_contract.validate(self.contract, self.prompt, self.tasks), [])

    def test_every_rule_of_the_seed_prompt_has_a_check_and_every_check_a_rule(self):
        in_prompt = set(put_contract.rule_ids(self.prompt))
        checked = {r["id"] for r in self.contract["rules"]}
        self.assertEqual(in_prompt - checked, set())
        self.assertEqual(checked - in_prompt, {"hidden_pass"})     # correctness is the task's, not a manual rule

    def test_a_rule_without_a_check_is_caught(self):
        prompt = self.prompt.replace('<rule id="scope">', '<rule id="scope">\n</rule>\n<rule id="unchecked_rule">')
        problems = put_contract.validate(self.contract, prompt, self.tasks)
        self.assertIn("prompt rule unchecked_rule has no check in contract.rules", problems)

    def test_the_split_is_six_and_four_and_covers_every_tag_twice(self):
        self.assertEqual(len(put_contract.tasks(INSTANCE, "train")), 6)
        self.assertEqual(len(put_contract.tasks(INSTANCE, "holdout")), 4)
        broken = [dict(t, tags=[x for x in t["tags"] if x != "cron"]) if t["split"] == "holdout" else t for t in self.tasks]
        self.assertIn("tag cron has no holdout task (coverage constraint)",
                      put_contract.validate(self.contract, self.prompt, broken))

    def test_weights_sum_to_one_and_the_deterministic_share_leads(self):
        w = self.contract["weights"]
        self.assertAlmostEqual(sum(w.values()), 1.0)
        judged = set(self.contract["judge"]["criteria"])
        self.assertAlmostEqual(sum(v for k, v in w.items() if k not in judged), 0.65)

    def test_the_seed_prompt_keeps_its_includes_and_size(self):
        body = self.prompt
        for inc in self.contract["prompt"]["required_includes"]:
            self.assertIn(f'<include file="{inc}"/>', body)
        self.assertLessEqual(len(body), self.contract["prompt"]["max_body_chars"])
        self.assertIn(self.contract["prompt"]["marker"], body)

    def test_the_thresholds_are_the_declared_ones(self):
        s = self.contract["statistics"]
        self.assertEqual((s["plateau_k"], s["obligatory_lower_bound"], s["min_gain_floor"]), (3, 0.90, 0.03))


if __name__ == "__main__":
    unittest.main()
