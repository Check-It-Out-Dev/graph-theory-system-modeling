"""The deterministic delta extractor against the real pack: an added Java file gets an entity row
with the heuristic type and its structural edges; a modified file is re-fingerprinted and its
structural out-edges replaced; a deleted file disappears with its edges; the churn threshold flips
the mode; pack.next rebuilds into a LadybugDB that answers a count."""

import csv
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "graph", "delta"))

import discover  # noqa: E402
import extract  # noqa: E402

PACK = os.path.join(R, "graph", "pack")
HAS_PACK = os.path.exists(os.path.join(PACK, "entities.csv"))
BE = "src/main/java/com/sm/instagram/platform/legal/"


class PureTests(unittest.TestCase):
    def test_eligibility_and_canon(self):
        self.assertTrue(extract.eligible("backend", "src/main/java/com/sm/X.java"))
        self.assertTrue(extract.eligible("backend", "src/main/resources/application.yml"))
        self.assertFalse(extract.eligible("backend", "target/classes/X.class"))
        self.assertFalse(extract.eligible("backend", "README.md"))
        self.assertTrue(extract.eligible("frontend", "src/app/x/x.component.ts"))
        self.assertFalse(extract.eligible("frontend", "src/app/api/generated.ts"))
        self.assertFalse(extract.eligible("frontend", "node_modules/x/index.js"))
        self.assertEqual(extract.canon("backend", "src/X.java"), "C:/Users/Norbert/IdeaProjects/checkItOut-be2/src/X.java")

    def test_entity_type_heuristic(self):
        et = extract.entity_type
        self.assertEqual(et("src/main/java/a/FooController.java", ""), "Actor")
        self.assertEqual(et("src/main/java/a/FooService.java", ""), "Process")
        self.assertEqual(et("src/main/java/a/Foo.java", "@Service class Foo {}"), "Process")
        self.assertEqual(et("src/main/java/a/FooRepository.java", ""), "Resource")
        self.assertEqual(et("src/main/java/a/AppConfig.java", ""), "Context")
        self.assertEqual(et("src/main/resources/application.yml", ""), "Context")
        self.assertEqual(et("src/test/java/a/FooTest.java", ""), "Rule")
        self.assertEqual(et("src/main/java/a/SecurityFilter.java", ""), "Rule")
        self.assertEqual(et("src/main/java/a/OrderEvent.java", ""), "Event")
        self.assertEqual(et("src/app/x/x.component.ts", ""), "Actor")
        self.assertEqual(et("src/app/x/x.service.ts", ""), "Process")
        self.assertEqual(et("src/app/x/x.spec.ts", ""), "Rule")

    def test_parse_changes_handles_renames(self):
        rows = extract.parse_changes("A\ta.java\nM\tb.java\nD\tc.java\nR100\told.java\tnew.java\n")
        self.assertEqual(rows, [("A", "a.java", None), ("M", "b.java", None), ("D", "c.java", None),
                                ("D", "old.java", None), ("A", "new.java", "old.java")])

    def test_fingerprint_is_content_only(self):
        self.assertEqual(extract.fingerprint("a\nb\n"), extract.fingerprint("a\nb\n"))
        self.assertTrue(extract.fingerprint("a\nb\n").startswith("size:4|lines:2|sha:"))


@unittest.skipUnless(HAS_PACK, "pack absent")
class DeltaTests(unittest.TestCase):
    def _repo(self):
        d = tempfile.mkdtemp(prefix="codemap-delta-")
        os.makedirs(os.path.join(d, BE))
        os.makedirs(os.path.join(d, "src/test/java/com/sm/instagram/platform/unit/service"))
        with open(os.path.join(d, BE, "ConsentReceiptService.java"), "w", encoding="utf-8") as f:
            f.write("package com.sm.instagram.platform.legal;\n"
                    "import com.sm.instagram.platform.legal.ConsentCookieService;\n"
                    "import org.springframework.stereotype.Service;\n"
                    "@Service\npublic class ConsentReceiptService extends BaseThing implements Runnable {\n"
                    "    private final LegalConsentService legal;\n    private final ConsentCookieService cookies;\n}\n")
        # a modified existing entity: the unit test of ConsentCookieService gets new content
        with open(os.path.join(d, "src/test/java/com/sm/instagram/platform/unit/service/ConsentCookieServiceUnitTest.java"), "w", encoding="utf-8") as f:
            f.write("package x;\nimport com.sm.instagram.platform.legal.ConsentCookieService;\nclass ConsentCookieServiceUnitTest {}\n")
        return d

    def test_added_modified_deleted_and_rebuild(self):
        repo = self._repo()
        out = tempfile.mkdtemp(prefix="codemap-delta-out-")
        changes = [("A", BE + "ConsentReceiptService.java", None),
                   ("M", "src/test/java/com/sm/instagram/platform/unit/service/ConsentCookieServiceUnitTest.java", None),
                   ("D", "src/test/java/com/sm/instagram/platform/unit/service/subscription/PaymentsDisabledBootGuardUnitTest.java", None),
                   ("A", "README.md", None)]
        delta = extract.run("backend", repo, changes, PACK, out, backlog_rows=[{"path": "backend/src/main/java/com/sm/nope/Missing.java"}], date="2026-09-17")
        cov = delta["coverage"]  # the fixture checkout holds only the changed files: every eligible one is indexed
        self.assertGreater(cov["eligible_files"], 0)
        self.assertEqual(cov["indexed_and_eligible"], cov["eligible_files"])
        self.assertEqual(cov["ratio"], 1.0)
        c = delta["counts"]
        self.assertEqual((delta["mode"], c["added"], c["modified"], c["deleted"]), ("delta", 1, 1, 1))
        self.assertEqual(c["after"], c["before"])
        added = delta["added"][0]
        self.assertEqual((added["name"], added["entity_type"]), ("ConsentReceiptService.java", "Process"))
        rels = {(e["dst"], e["rel"]) for e in added["edges_out"]}
        self.assertIn(("ConsentCookieService.java", "IMPORTS"), rels)
        self.assertIn(("LegalConsentService.java", "INJECTS"), rels)
        self.assertIn(("ConsentCookieService.java", "INJECTS"), rels)
        self.assertEqual(delta["unassigned"], [added["file_path"]])
        # pack.next
        nxt = os.path.join(out, "pack.next")
        ents = list(csv.DictReader(open(os.path.join(nxt, "entities.csv"), encoding="utf-8")))
        names = {e["name"] for e in ents}
        self.assertIn("ConsentReceiptService.java", names)
        self.assertNotIn("PaymentsDisabledBootGuardUnitTest.java", names)
        new = next(e for e in ents if e["name"] == "ConsentReceiptService.java")
        self.assertEqual((new["subsystem"], new["delta_batch"], new["layer"]), ("", "2026-09-17", "Process"))
        edges = list(csv.DictReader(open(os.path.join(nxt, "edges.csv"), encoding="utf-8")))
        self.assertFalse(any("PaymentsDisabledBootGuardUnitTest.java" in (e["src"], e["dst"]) for e in edges))
        self.assertTrue(any(e["src"] == "ConsentCookieService.java" and e["dst"] == "ConsentCookieServiceUnitTest.java" and e["rel"] == "TESTED_BY" for e in edges))
        mod = next(e for e in ents if e["name"] == "ConsentCookieServiceUnitTest.java")
        self.assertEqual(mod["delta_batch"], "2026-09-17")
        self.assertTrue(mod["fingerprint"].startswith("size:"))
        # the LadybugDB answers
        import real_ladybug as lb
        conn = lb.Connection(lb.Database(os.path.join(nxt, "codemap.lbdb"), read_only=True))
        n = conn.execute("MATCH (e:Entity) RETURN count(*)").get_next()[0]
        self.assertEqual(n, len(ents))
        r = conn.execute("MATCH (a:Entity)-[d:Dep]->(b:Entity) WHERE a.name = 'ConsentReceiptService.java' RETURN b.name, d.rel")
        got = set()
        while r.has_next():
            got.add(tuple(r.get_next()))
        self.assertIn(("ConsentCookieService.java", "IMPORTS"), got)
        del conn

    def test_churn_flips_to_full(self):
        repo = self._repo()
        out = tempfile.mkdtemp(prefix="codemap-delta-out-")
        ents = list(csv.DictReader(open(os.path.join(PACK, "entities.csv"), encoding="utf-8")))
        many = [("D", e["file_path"].split("checkItOut-be2/", 1)[1], None) for e in ents if "checkItOut-be2/" in e["file_path"]][:200]
        delta = extract.run("backend", repo, many, PACK, out, date="2026-09-17")
        self.assertEqual(delta["mode"], "full")
        self.assertGreater(delta["churn"], 0.1)
        self.assertFalse(os.path.exists(os.path.join(out, "pack.next")))


class DiscoverTests(unittest.TestCase):
    def test_dispatch_row_and_poll_skips_seen_and_forks(self):
        p = discover.plan("workflow_dispatch", repo="backend", sha="abc123", base="base9", pr="7", http=lambda path: {}, seen_heads=set())
        self.assertEqual(p["include"], [{"repo": "backend", "public": "Check-It-Out-Dev/checkitout-backend", "sha": "abc123", "base": "base9", "pr": "7"}])

        def http(path):
            if "/pulls" in path:
                return [{"number": 1, "head": {"sha": "h1", "repo": {"full_name": "Check-It-Out-Dev/checkitout-backend"}}, "base": {"sha": "b1"}},
                        {"number": 2, "head": {"sha": "h2", "repo": {"full_name": "someone/fork"}}, "base": {"sha": "b1"}},
                        {"number": 3, "head": {"sha": "seen", "repo": {"full_name": "Check-It-Out-Dev/checkitout-backend"}}, "base": {"sha": "b1"}}]                     if "backend" in path else []
            if path.endswith("/commits/main"):
                return {"sha": "mainsha"}
            return {}
        p = discover.plan("schedule", http=http, seen_heads={"seen", "mainsha"})
        shas = [(r["repo"], r["sha"], r["pr"]) for r in p["include"]]
        self.assertIn(("backend", "h1", "1"), shas)
        self.assertNotIn(("backend", "h2", "2"), shas)
        self.assertNotIn(("backend", "seen", "3"), shas)
        self.assertFalse(any(r["sha"] == "mainsha" for r in p["include"]))


if __name__ == "__main__":
    unittest.main()
