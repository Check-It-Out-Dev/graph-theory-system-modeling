"""The statistics of the prompt-under-test pipeline against values computed by hand or from the literature."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "eval", "put"))

import put_stats as s  # noqa: E402


class StatsTests(unittest.TestCase):
    def test_wilson_matches_the_textbook_interval(self):
        lo, hi = s.wilson(8, 10)                                   # Newcombe (1998) method 3: 0.4902 to 0.9433
        self.assertAlmostEqual(lo, 0.4902, places=3)
        self.assertAlmostEqual(hi, 0.9433, places=3)
        self.assertEqual(s.wilson(0, 0), (0.0, 1.0))

    def test_zero_failures_need_35_runs_for_the_090_bound(self):
        self.assertEqual(s.min_n_for_bound(0.90), 35)             # n / (n + 1.96^2) >= 0.90 first at n = 35
        self.assertTrue(s.obligatory(35, 35))
        self.assertFalse(s.obligatory(34, 34))
        self.assertFalse(s.obligatory(39, 40))                    # one failure in 40 is not enough

    def test_delta_takes_the_larger_noise(self):
        d, agent = s.delta(0.02, 0.10, tasks=6, k=3)
        self.assertAlmostEqual(agent, 1.96 * 0.10 * (2 / 18) ** 0.5)
        self.assertAlmostEqual(d, agent)
        self.assertEqual(s.delta(0.2, 0.01, 6, 3)[0], 0.2)

    def test_pooled_sd_ignores_single_values(self):
        self.assertAlmostEqual(s.pooled_sd([[1, 3], [2, 2, 2], [5]]), (2 / 3) ** 0.5)

    def test_bootstrap_is_seeded_and_brackets_the_point(self):
        a = {"t1": [0.5, 0.6, 0.55], "t2": [0.4, 0.45, 0.5], "t3": [0.7, 0.72, 0.68]}
        b = {"t1": [0.7, 0.75, 0.72], "t2": [0.6, 0.62, 0.58], "t3": [0.8, 0.82, 0.79]}
        point, lo, hi = s.paired_bootstrap(a, b, resamples=2000)
        self.assertEqual((point, lo, hi), s.paired_bootstrap(a, b, resamples=2000))
        self.assertLess(lo, point)
        self.assertLess(point, hi)
        self.assertGreater(lo, 0)

    def test_sign_test_floor_with_four_tasks(self):
        self.assertEqual(s.sign_test([0.1, 0.2, 0.05, 0.3]), (4, 4, 0.0625))
        self.assertEqual(s.sign_test([0.1, -0.2, 0.0]), (1, 2, 0.75))

    def test_kappa_and_ac1_under_a_skewed_prevalence(self):
        # the D-R18 paradox: 88 % agreement with a 90 % yes-rate gives a low kappa and a high AC1
        judge = [1] * 29 + [0] * 5
        oracle = [1] * 28 + [0] + [1] * 3 + [0] * 2
        k = s.kappa(judge, oracle)
        self.assertAlmostEqual(k["agreement"], 30 / 34)            # 28 yes-yes + 2 no-no
        self.assertLess(k["kappa"], 0.6)
        self.assertGreater(s.ac1(judge, oracle), 0.8)

    def test_perfect_agreement(self):
        self.assertEqual(s.kappa([1, 0, 1], [1, 0, 1])["kappa"], 1.0)
        self.assertEqual(s.ac1([1, 0, 1], [1, 0, 1]), 1.0)

    def test_spearman_with_ties(self):
        self.assertAlmostEqual(s.spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(s.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertGreater(s.spearman([5, 4, 4, 2, 1], [5, 5, 4, 3, 1]), 0.9)


if __name__ == "__main__":
    unittest.main()
