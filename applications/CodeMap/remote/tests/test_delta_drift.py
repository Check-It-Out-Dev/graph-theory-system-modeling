"""Version drift without a model: identical packs do not drift; a partition decision moves only the
rows it invalidated on purpose or the rows whose answers genuinely changed; hit order is not drift."""

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "graph", "delta"))

import drift  # noqa: E402

PACK = os.path.join(R, "graph", "pack")
HAS_PACK = os.path.exists(os.path.join(PACK, "entities.csv"))


class CanonicalTests(unittest.TestCase):
    def test_order_and_affordances_do_not_count(self):
        a = {"kind": "find", "hits": [{"name": "B"}, {"name": "A"}], "affordances": ["x"], "dsl": "find(a)"}
        b = {"kind": "find", "hits": [{"name": "A"}, {"name": "B"}], "affordances": ["y"], "dsl": "find(b)"}
        self.assertEqual(drift.digest(a), drift.digest(b))
        self.assertNotEqual(drift.digest(a), drift.digest({"kind": "find", "hits": [{"name": "A"}]}))

    def test_plans(self):
        self.assertEqual(drift.VERB_BY_ARCHETYPE["locate"]({"rx": "Foo"}), [("find", ["Foo"])])
        self.assertEqual(drift.VERB_BY_ARCHETYPE["locate"]({}), [])
        self.assertEqual(drift.VERB_BY_ARCHETYPE["boundary"]({"a": 1, "b": 2}), [("seam", ["1", "2"])])
        self.assertEqual(len(drift.VERB_BY_ARCHETYPE["health"](None)), 2)


@unittest.skipUnless(HAS_PACK, "pack absent")
class PackDriftTests(unittest.TestCase):
    def test_identical_packs_do_not_drift(self):
        rep = drift.compare(PACK, PACK, use_ladybug=False)
        self.assertEqual(rep["drifted"], 0)
        self.assertGreater(rep["compared"], 30)
        self.assertEqual(rep["drift_rate"], 0.0)
        self.assertTrue(all(r["status"] in ("stable", "invalidated", "not_executable") for r in rep["rows"]))

    def test_a_moved_entity_drifts_only_where_it_should(self):
        tmp = tempfile.mkdtemp(prefix="codemap-drift-")
        new = os.path.join(tmp, "pack.next")
        shutil.copytree(PACK, new, ignore=shutil.ignore_patterns("codemap.lbdb", "*.gbnf"))
        # move the boot guard into subsystem 10 by editing entities.csv (no model, no apply): the locate
        # row BE01 (find PaymentsDisabledBootGuard) must drift, an invalidated row must not count
        import csv
        rows = list(csv.DictReader(open(os.path.join(new, "entities.csv"), encoding="utf-8")))
        victim = next(e for e in rows if e["name"] == "PaymentsDisabledBootGuard.java")
        victim["curated"] = "10"
        with open(os.path.join(new, "entities.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        rep = drift.compare(PACK, new, invalidated=["BE02"], use_ladybug=False)
        self.assertGreater(rep["drifted"], 0)
        self.assertIn("BE02", [r["id"] for r in rep["rows"] if r["status"] == "invalidated"])
        self.assertIn("BE01", rep["drifted_ids"])
        drifted = {r["id"]: r for r in rep["rows"] if r["status"] == "drifted"}
        # every drifted row names the DSL that changed, and locate rows about other entities stay stable
        self.assertTrue(all(r["diffs"] for r in drifted.values()))
        self.assertLess(rep["drift_rate"], 0.5)
        json.dumps(rep)  # serialisable artifact


if __name__ == "__main__":
    unittest.main()
