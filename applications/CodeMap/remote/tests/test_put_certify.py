"""Certification and promotion without a model: the hold-out verdict, the ceiling test that names the rules a prompt
could not teach, and the promotion gate, on synthetic run records."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "eval", "put"))

import put_certify  # noqa: E402
import put_contract  # noqa: E402
import put_promote  # noqa: E402

INSTANCE = "backend-conventions"
UNIVERSAL = ("graph_first", "exemplar_read", "feature_first", "constructor_injection", "tests_written", "build_green",
             "scope", "marker", "hidden_pass")


def records(scores_by_task, split_of, reps, rule_pass=lambda task, rep: True, rules=UNIVERSAL):
    out = []
    for task, base in scores_by_task.items():
        for r in range(reps):
            checks = {rid: {"value": 1.0 if rule_pass(task, r) or rid != "graph_first" else 0.0, "applicable": True,
                            "passed": rule_pass(task, r) or rid != "graph_first", "seen": ""} for rid in rules}
            out.append({"task": task, "split": split_of[task], "rep": r, "score": base + 0.01 * (r % 2), "checks": checks})
    return out


class CertifyTests(unittest.TestCase):
    def setUp(self):
        self.contract = put_contract.load(INSTANCE)
        self.split = {t["id"]: t["split"] for t in put_contract.tasks(INSTANCE)}
        self.train = [t for t, s in self.split.items() if s == "train"]
        self.hold = [t for t, s in self.split.items() if s == "holdout"]

    def test_a_clear_holdout_gain_works(self):
        seed = records({t: 0.60 for t in self.split}, self.split, 3)
        cand = records({t: 0.75 for t in self.split}, self.split, 4)
        v, _, rules_c = put_certify.verdict(seed, cand, self.contract, delta=0.03)
        self.assertTrue(v["works"], v["why"])
        self.assertGreater(v["gain_holdout"]["ci"][0], 0)
        reason, untaught, _ = put_certify.ceiling(rules_c, self.contract)
        self.assertEqual(reason, "saturated_at_ceiling")              # 40 clean runs per universal rule
        self.assertEqual(untaught, [])

    def test_a_gain_inside_the_noise_does_not_work(self):
        seed = records({t: 0.60 for t in self.split}, self.split, 3)
        cand = records({t: 0.62 for t in self.split}, self.split, 4)
        v, _, _ = put_certify.verdict(seed, cand, self.contract, delta=0.05)
        self.assertFalse(v["works"])
        self.assertIn("not above delta", v["why"])

    def test_a_rule_below_the_bound_is_named_untaught_with_its_enforcement(self):
        cand = records({t: 0.8 for t in self.split}, self.split, 4, rule_pass=lambda task, rep: rep != 0)   # graph_first 30/40
        _, _, rules_c = put_certify.verdict(records({t: 0.6 for t in self.split}, self.split, 3), cand, self.contract, 0.03)
        reason, untaught, _ = put_certify.ceiling(rules_c, self.contract)
        self.assertEqual(reason, "saturated_below_ceiling")
        self.assertEqual([u["rule"] for u in untaught], ["graph_first"])
        self.assertIn("hook", untaught[0]["enforce"])

    def test_losing_an_obligatory_rule_blocks_the_verdict(self):
        seed = records({t: 0.6 for t in self.split}, self.split, 4)                                      # 40 clean
        cand = records({t: 0.9 for t in self.split}, self.split, 4, rule_pass=lambda task, rep: rep != 0)
        v, _, _ = put_certify.verdict(seed, cand, self.contract, delta=0.03)
        self.assertFalse(v["works"])
        self.assertEqual(v["obligatory_lost"], ["graph_first"])


class PromoteTests(unittest.TestCase):
    def setUp(self):
        self.contract = put_contract.load(INSTANCE)

    def cert(self, mean, lo, lost=()):
        recs = [{"task": f"h{i}", "split": "holdout", "rep": r} for i in range(4) for r in range(4)]
        return {"verdict": {"gain_holdout": {"mean": mean, "ci": [lo, mean + 0.1], "delta": 0.02}, "obligatory_lost": list(lost)},
                "records": recs}

    def test_the_gate_opens_on_a_certified_gain(self):
        ok, reasons = put_promote.decide(self.cert(0.08, 0.02), self.contract, {"stop": {"reason": "plateau"}, "candidates": [{}]}, "a", "b")
        self.assertTrue(ok, reasons)

    def test_the_gate_names_each_failed_condition(self):
        ok, reasons = put_promote.decide(self.cert(0.02, -0.01, ["scope"]), self.contract, {"stop": {"reason": "owner"}}, "same", "same")
        self.assertFalse(ok)
        text = " ".join(reasons)
        for part in ("not above", "CI lower end", "obligatory rules lost", "stopped by owner", "the candidate is the current prompt"):
            self.assertIn(part, text)

    def test_the_distributed_file_says_where_it_came_from_and_works_without_the_graph(self):
        body = put_contract.prompt_text(INSTANCE, "v1")
        text = put_promote.render_for_repo(body, INSTANCE, "certify-x")
        self.assertIn("certified by run certify-x", text)
        self.assertIn("no `graph` tool", text)
        self.assertNotIn("<include file=", text)


if __name__ == "__main__":
    unittest.main()
