"""Quality rates replayed from committed fixtures: the properties every night must satisfy, the
expected.json snapshot (mutation: flip one terminal and the grounded rate moves), gains pairing."""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "eval", "quality"))

import quality  # noqa: E402

FIX = os.path.join(R, "telemetry", "fixtures")


def fixture_events():
    return [json.loads(l) for l in open(os.path.join(FIX, "events.sample.jsonl"), encoding="utf-8") if l.strip()]


def fixture_judge():
    return json.load(open(os.path.join(FIX, "judge.sample.json"), encoding="utf-8"))


def fixture_humans():
    return [json.loads(l) for l in open(os.path.join(FIX, "humans.sample.jsonl"), encoding="utf-8") if l.strip()]


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = quality.compute("2025-09-16", fixture_events(), fixture_judge(), fixture_humans())

    def test_matches_the_committed_expectation(self):
        expected = json.load(open(os.path.join(R, "eval", "quality", "fixtures", "expected.json"), encoding="utf-8"))
        got = {k: v for k, v in self.doc.items() if k != "date"}
        exp = {k: v for k, v in expected.items() if k != "date"}
        self.assertEqual(got, exp)

    def test_properties(self):
        d = self.doc
        for k in ("codemap_grounded_rate", "codemap_correct_rate", "codemap_rating_ge4_rate", "codemap_cache_read_ratio",
                  "codemap_pointer_verified_rate", "codemap_oracle_success_rate"):
            self.assertTrue(0.0 <= d[k] <= 1.0, k)
        self.assertEqual(sum(d["codemap_requests_total"].values()), d["n"]["asks"])
        self.assertTrue(all(v >= 0 for v in d["codemap_credits_total"].values()))
        self.assertGreater(d["codemap_credits_per_correct_answer"], 0)
        tok = d["codemap_tokens_total"]
        self.assertGreater(tok["cached"], tok["prompt"])
        self.assertEqual(d["codemap_gain"]["n_pairs"], 2)

    def test_permutation_invariant_and_idempotent(self):
        evs = fixture_events()
        a = quality.compute("x", evs, fixture_judge(), fixture_humans())
        b = quality.compute("x", list(reversed(evs)), fixture_judge(), fixture_humans())
        self.assertEqual(a, b)
        self.assertEqual(a, quality.compute("x", evs, fixture_judge(), fixture_humans()))

    def test_flipping_a_terminal_moves_the_rates(self):
        jd = fixture_judge()
        first = next(r for r in jd["rows"] if r["judge"]["grounded"] >= 4)
        first["judge"]["grounded"] = 1
        d2 = quality.compute("x", fixture_events(), jd, fixture_humans())
        self.assertNotEqual(d2["codemap_grounded_rate"], self.doc["codemap_grounded_rate"])

    def test_flatten_is_influx_lines_without_currency(self):
        lines = quality.flatten(self.doc, "2025-09-16")
        self.assertTrue(any(l.startswith("codemap_quality_grounded_rate,night=2025-09-16 value=") for l in lines))
        self.assertTrue(any(l.startswith("codemap_quality_gain,night=2025-09-16,key=tokens_ratio_mean") for l in lines))
        self.assertFalse(any("usd" in l.lower() for l in lines))

    def test_gains_need_pairs(self):
        g = quality.gains([{"persona": "p", "seed_id": "a", "mode": "codemap", "usage": {"input_tokens": 10}, "num_turns": 3}])
        self.assertEqual((g["n_pairs"], g["tokens_ratio_mean"]), (0, None))
        # a baseline whose partner was skipped (budget) is no pair, and neither is a skipped baseline
        rows = [{"persona": "p", "seed_id": "a", "mode": "baseline", "usage": {"input_tokens": 10}, "num_turns": 3},
                {"persona": "p", "seed_id": "a", "mode": "codemap", "skipped": "budget_exhausted"},
                {"persona": "p", "seed_id": "b", "mode": "baseline", "skipped": "no_budget_for_partner"},
                {"persona": "p", "seed_id": "b", "mode": "codemap", "skipped": "baseline_skipped"},
                {"persona": "p", "seed_id": "c", "mode": "baseline", "usage": {"input_tokens": 20}, "num_turns": 4},
                {"persona": "p", "seed_id": "c", "mode": "codemap", "usage": {"input_tokens": 10}, "num_turns": 2}]
        g = quality.gains(rows)
        self.assertEqual((g["n_pairs"], g["baselines"], g["tokens_ratio_mean"], g["turns_delta_mean"]), (1, 2, 2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
