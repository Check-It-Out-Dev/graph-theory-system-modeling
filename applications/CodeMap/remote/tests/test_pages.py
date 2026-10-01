"""The quality page is built from committed artifacts alone and says the same numbers they do."""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "tools", "pages"))

import build_quality  # noqa: E402


class CampaignTests(unittest.TestCase):
    def test_pairs_by_persona_and_seed_across_nights(self):
        sys.path.insert(0, os.path.join(R, "eval", "quality"))
        import campaign
        rows = [{"persona": "p", "seed_id": "BE01", "mode": "baseline", "usage": {"input_tokens": 100, "output_tokens": 10}, "num_turns": 12, "duration_ms": 60000,
                 "report": {"conversations": [{"turns": [{"rating": 3}]}]}},
                {"persona": "p", "seed_id": "BE01", "mode": "codemap", "usage": {"input_tokens": 50, "output_tokens": 10}, "num_turns": 8, "duration_ms": 30000,
                 "report": {"conversations": [{"turns": [{"rating": 5}]}]}},
                {"persona": "p", "seed_id": "BE02", "mode": "baseline", "usage": {}, "num_turns": 3}]
        pairs = campaign.pairs_of(rows, "n1")
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0]["tokens_ratio"], pairs[0]["turns_delta"], pairs[0]["seconds_delta"]), (round(110 / 60, 3), 4, 30.0))
        self.assertEqual((pairs[0]["rating_baseline"], pairs[0]["rating_codemap"]), (3, 5))
        doc = campaign.campaign(os.path.join(R, "eval", "humans", "runs"))
        self.assertIn("n_pairs", doc)
        self.assertEqual(doc["gated"], doc["n_pairs"] >= 30)


class PagesTests(unittest.TestCase):
    def test_collect_reads_the_nights_and_decisions(self):
        data = build_quality.collect(R)
        self.assertGreaterEqual(len(data["nights"]), 1)
        night = next(n for n in data["nights"] if n["date"] == "2026-09-16")
        q = json.load(open(os.path.join(R, "eval", "quality", "runs", "2026-09-16.json"), encoding="utf-8"))
        self.assertEqual(night["grounded"], q["codemap_grounded_rate"])
        self.assertEqual(night["kappa"].get("oracle"), q["codemap_judge_kappa"]["oracle"])
        self.assertGreaterEqual(len(data["decisions"]), 1)
        d = data["decisions"][0]
        self.assertEqual((d["version"], d["repo"], d["kind"]), ("1.0.1", "backend", "accept"))
        self.assertEqual(d["assignments"], 7)
        self.assertGreaterEqual(len(data["dashboards"]), 6)
        self.assertTrue(all(x["url"].startswith("https://checkitoutapp.grafana.net/public-dashboards/") for x in data["dashboards"]))

    def test_build_writes_html_data_and_proofs(self):
        out = tempfile.mkdtemp(prefix="codemap-pages-")
        data = build_quality.build(out, R)
        html = open(os.path.join(out, "index.html"), encoding="utf-8").read()
        self.assertIn("<title>CodeMap Remote — quality</title>", html)
        latest = max(data["nights"], key=lambda n: n["date"])
        self.assertIn(build_quality.pct(latest["grounded"]), html)  # the latest night's grounded rate is on the page
        self.assertIn("pack", html)
        self.assertIn("codemap.checkitout.app/mcp", html)
        self.assertIn("prefers-color-scheme", html)
        self.assertNotIn("$", html.split("<footer>")[0].split("Credits")[0][-200:] if "Credits" in html else "")  # no currency near the credits
        self.assertTrue(os.path.exists(os.path.join(out, "data.json")))
        for name in data["proofs"]:
            self.assertTrue(os.path.exists(os.path.join(out, "proofs", name)))

    def test_sparkline_is_honest_about_one_point(self):
        svg = build_quality.sparkline([0.9])
        self.assertEqual(svg.count("<circle"), 1)
        self.assertIn("no data", build_quality.sparkline([None, None]))
        self.assertEqual(build_quality.pct(None), "—")
        self.assertEqual(build_quality.pct(0.9143), "91 %")


if __name__ == "__main__":
    unittest.main()
