"""Dashboards as code: the builder is deterministic, the lint catches the two classic public-dashboard
breakages (a template variable, a duplicate panel id) and a bare counter query."""

import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "tools", "grafana"))

import build_dashboards  # noqa: E402
import lint  # noqa: E402


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = [d.build() for d in build_dashboards.build_all()]

    def test_six_dashboards_lint_clean(self):
        self.assertEqual(len(self.docs), 6)
        for d in self.docs:
            self.assertEqual(lint.lint_doc(d, d["uid"]), [], d["uid"])
            self.assertEqual(d["templating"]["list"], [])
            ids = [p["id"] for p in d["panels"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertLessEqual(max(p["gridPos"]["x"] + p["gridPos"]["w"] for p in d["panels"]), 24)

    def test_builder_is_deterministic(self):
        a = json.dumps([d.build() for d in build_dashboards.build_all()], sort_keys=True)
        b = json.dumps([d.build() for d in build_dashboards.build_all()], sort_keys=True)
        self.assertEqual(a, b)

    def test_breakage_template_variable(self):
        d = copy.deepcopy(self.docs[0])
        d["templating"]["list"] = [{"name": "user", "type": "query"}]
        self.assertTrue(any("template" in p for p in lint.lint_doc(d, "x")))

    def test_breakage_duplicate_panel_id(self):
        d = copy.deepcopy(self.docs[0])
        d["panels"][2]["id"] = d["panels"][1]["id"]
        self.assertTrue(any("duplicate" in p for p in lint.lint_doc(d, "x")))

    def test_breakage_bare_counter_and_currency(self):
        d = copy.deepcopy(self.docs[1])
        stat = next(p for p in d["panels"] if p["type"] == "stat")
        stat["targets"][0]["expr"] = "codemap_feedback_total"
        probs = lint.lint_doc(d, "x")
        self.assertTrue(any("without max_over_time" in p for p in probs))
        d2 = copy.deepcopy(self.docs[3])
        text = next(p for p in d2["panels"] if p["type"] == "text")
        text["options"]["content"] += "\n\nAbout $3 per answer in USD."
        self.assertTrue(any("currency" in p for p in lint.lint_doc(d2, "x")))

    def test_committed_files_match_the_builder(self):
        out = os.path.join(R, "observability", "grafana")
        for d in self.docs:
            p = os.path.join(out, f"{d['uid']}.json")
            self.assertTrue(os.path.exists(p), p)
            self.assertEqual(json.load(open(p, encoding="utf-8")), d)


if __name__ == "__main__":
    unittest.main()
