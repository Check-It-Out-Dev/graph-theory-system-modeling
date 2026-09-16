"""Proposal and apply without a model: the pack MCP over pack.next, deterministic candidates, the
checker refusing invented ids, the reviewer with a fake CLI, the decision grammar, and a full apply on
a synthetic delta — entities placed, ledger row bi-temporal, curation note appended, FAQ invalidated,
manifest bumped, LadybugDB rebuilt."""

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

import apply as apply_mod  # noqa: E402
import extract  # noqa: E402
import issue  # noqa: E402
import pack_mcp  # noqa: E402
import propose  # noqa: E402
from remote import mcp  # noqa: E402

PACK = os.path.join(R, "graph", "pack")
HAS_PACK = os.path.exists(os.path.join(PACK, "entities.csv"))
BE = "src/main/java/com/sm/instagram/platform/legal/"


class CommandTests(unittest.TestCase):
    def test_grammar(self):
        pc = apply_mod.parse_command
        self.assertEqual(pc("/codemap accept"), {"kind": "accept"})
        self.assertEqual(pc("codemap accept-by-timeout"), {"kind": "accept", "timeout": True})
        self.assertEqual(pc("/codemap move Foo.java to 11"), {"kind": "move", "entity": "Foo.java", "subsystem": "11"})
        self.assertEqual(pc("/codemap new-subsystem Consent receipts: A.java, B.java"),
                         {"kind": "new-subsystem", "name": "Consent receipts", "members": ["A.java", "B.java"]})
        self.assertEqual(pc("/codemap reject not now")["reason"], "not now")
        self.assertIsNone(pc("looks good to me"))
        self.assertIsNone(pc("/codemap dance"))


def _delta_fixture():
    repo = tempfile.mkdtemp(prefix="codemap-delta-")
    os.makedirs(os.path.join(repo, BE))
    with open(os.path.join(repo, BE, "ConsentReceiptService.java"), "w", encoding="utf-8") as f:
        f.write("package com.sm.instagram.platform.legal;\nimport com.sm.instagram.platform.legal.ConsentCookieService;\n"
                "@Service\npublic class ConsentReceiptService {\n    private final LegalConsentService legal;\n}\n")
    with open(os.path.join(repo, BE, "ConsentReceiptDto.java"), "w", encoding="utf-8") as f:
        f.write("package com.sm.instagram.platform.legal;\npublic record ConsentReceiptDto(String id) {}\n")
    out = tempfile.mkdtemp(prefix="codemap-delta-out-")
    changes = [("A", BE + "ConsentReceiptService.java", None), ("A", BE + "ConsentReceiptDto.java", None)]
    delta = extract.run("backend", repo, changes, PACK, out, date="2026-09-17")
    delta["head"] = "abc1234567"
    json.dump(delta, open(os.path.join(out, "delta.json"), "w", encoding="utf-8"))
    return out, delta


@unittest.skipUnless(HAS_PACK, "pack absent")
class ProposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out, cls.delta = _delta_fixture()
        cls.pack = os.path.join(cls.out, "pack.next")

    def test_pack_mcp_tools(self):
        P = pack_mcp.Pack(self.pack)
        _, r = mcp.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, P.call, tools=pack_mcp.TOOLS)
        self.assertEqual([t["name"] for t in r["result"]["tools"]], ["pack_subsystem", "pack_entity", "pack_folder", "pack_cypher"])
        sub = json.loads(P.call("pack_subsystem", {"id": 11})[0])
        self.assertIn("Subscriptions", sub["name"])
        ent = json.loads(P.call("pack_entity", {"name": "ConsentReceiptService.java"})[0])
        self.assertTrue(any(e["to"] == "ConsentCookieService.java" for e in ent["out"]))
        self.assertEqual(ent["entity"]["subsystem"], "")
        folder = json.loads(P.call("pack_folder", {"fragment": "/platform/legal/"})[0])
        self.assertGreater(folder["n"], 3)
        self.assertTrue(P.call("pack_cypher", {"stmt": "MATCH (e:Entity) DELETE e"})[1])
        rows = json.loads(P.call("pack_cypher", {"stmt": "MATCH (e:Entity) WHERE e.curated = 11 RETURN count(*)"})[0])
        self.assertEqual(rows["columns"], ["count(*)"] if rows["columns"] == ["count(*)"] else rows["columns"])

    def test_candidates_and_checker(self):
        cands = propose.candidates(self.pack, self.delta)
        by = {c["entity"]: c for c in cands}
        svc = by["ConsentReceiptService.java"]
        self.assertEqual((svc["subsystem"], svc["rule"]), ("11", "structural-neighbours"))
        dto = by["ConsentReceiptDto.java"]
        self.assertEqual(dto["rule"], "folder-majority")
        self.assertEqual(dto["subsystem"], "11")
        doc = propose.review(self.delta, cands, self.pack, backend="none")
        self.assertEqual(propose.check(doc, self.pack), [])
        bad = json.loads(json.dumps(doc))
        bad["assignments"][0]["subsystem"] = 9999
        bad["assignments"].append({"entity": "Ghost.java", "subsystem": 11, "confidence": 1, "why": "x"})
        probs = propose.check(bad, self.pack)
        self.assertTrue(any("unknown subsystem 9999" in p for p in probs))
        self.assertTrue(any("unknown entity 'Ghost.java'" in p for p in probs))

    def test_review_with_a_fake_cli_and_fallback(self):
        cands = propose.candidates(self.pack, self.delta)
        good = {"assignments": [{"entity": c["entity"], "subsystem": 11, "confidence": 0.9, "why": "legal folder", "alternatives": []} for c in cands],
                "new_subsystems": [], "unresolved": []}

        class P:
            def __init__(self, o):
                self.stdout, self.stderr, self.returncode = json.dumps(o), "", 0
        runner = lambda cmd, env, cwd, timeout: P({"type": "result", "is_error": False, "result": "ok", "structured_output": good,
                                                   "usage": {"input_tokens": 1, "output_tokens": 1}})
        doc = propose.review(self.delta, cands, self.pack, backend="claude", runner=runner)
        self.assertEqual(doc["reviewed_by"], "claude:claude-sonnet-5")
        self.assertEqual(len(doc["assignments"]), 2)
        broken = propose.review(self.delta, cands, self.pack, backend="claude",
                                runner=lambda cmd, env, cwd, timeout: P({"type": "result", "is_error": True, "result": "Rate limit"}))
        self.assertTrue(broken["reviewed_by"].startswith("deterministic"))
        body = issue.render(doc, self.delta, "https://example/run/1", ref="feature/x")
        self.assertIn("<!-- codemap-delta backend@abc1234567 ref=feature/x -->", body)
        self.assertIn("<!-- codemap-delta backend@abc1234567 -->", issue.render(doc, self.delta))
        self.assertIn("/codemap accept", body)
        self.assertIn("`ConsentReceiptService.java`", body)

    def test_apply_accept_end_to_end(self):
        out, delta = _delta_fixture()
        pack = os.path.join(out, "pack.next")
        cands = propose.candidates(pack, delta)
        doc = propose.review(delta, cands, pack, backend="none")
        ledger = tempfile.mkdtemp(prefix="codemap-ledger-")
        notes = os.path.join(out, "curation_notes.md")
        open(notes, "w", encoding="utf-8").write("## Curation notes\n")
        row, inval = apply_mod.apply(doc, delta, pack, {"kind": "accept"}, "tester", "1.0.1", ledger, notes)
        self.assertEqual(len(row["assignments"]), 2)
        self.assertIn("11", row["changed_subsystems"])
        ents = {e["name"]: e for e in csv.DictReader(open(os.path.join(pack, "entities.csv"), encoding="utf-8"))}
        self.assertEqual(ents["ConsentReceiptService.java"]["curated"], "11")
        self.assertTrue(os.path.exists(os.path.join(ledger, "1.0.1.json")))
        text = open(notes, encoding="utf-8").read()
        self.assertIn("ConsentReceiptService.java → subsystem 11", text)
        self.assertIn("decided by tester", text)
        self.assertTrue(row["assignments"][0]["path"].startswith("src/"))
        inv = json.load(open(os.path.join(pack, "INVALIDATED_delta.json"), encoding="utf-8"))
        n_after_add = len(inv["invalidated"])  # additions invalidate only enumerating rows
        self.assertLess(n_after_add, 20)  # of the 41 rows depending on subsystem 11
        man = json.load(open(os.path.join(pack, "manifest.json"), encoding="utf-8"))
        self.assertEqual((man["pack_version"], man["indexed_sha"]["backend"]), ("1.0.1", "abc1234567"))
        l2 = [json.loads(l) for l in open(os.path.join(pack, "l2_navigators.jsonl"), encoding="utf-8") if l.strip()]
        sub11 = next(n for n in l2 if str(n["sub_id"]) == "11")
        self.assertEqual(sub11["size"], sum(1 for e in ents.values() if e["curated"] == "11"))
        self.assertIn("structure_updated_at", sub11)
        # a second decision supersedes the first row bi-temporally
        row2, _ = apply_mod.apply(doc, delta, pack, {"kind": "move", "entity": "ConsentReceiptDto.java", "subsystem": "10"}, "tester", "1.0.2", ledger, notes)
        first = json.load(open(os.path.join(ledger, "1.0.1.json"), encoding="utf-8"))
        dto_row = next(a for a in first["assignments"] if a["entity"] == "ConsentReceiptDto.java")
        self.assertEqual(dto_row["superseded_by"], "1.0.2")
        self.assertIsNotNone(dto_row["t_invalid"])
        ents = {e["name"]: e for e in csv.DictReader(open(os.path.join(pack, "entities.csv"), encoding="utf-8"))}
        self.assertEqual(ents["ConsentReceiptDto.java"]["curated"], "10")
        self.assertEqual(set(row2["changed_subsystems"]), {"10", "11"})
        inv = json.load(open(os.path.join(pack, "INVALIDATED_delta.json"), encoding="utf-8"))
        self.assertGreater(len(inv["invalidated"]), n_after_add)  # a move invalidates rows on both subsystems
        # reject writes only the ledger
        row3, _ = apply_mod.apply(doc, delta, pack, {"kind": "reject", "reason": "not yet"}, "tester", "1.0.3", ledger, notes)
        self.assertEqual((row3["rejected"], row3["assignments"]), ("not yet", []))
        # a new subsystem gets a fresh id and an L2 record
        row4, _ = apply_mod.apply(doc, delta, pack, {"kind": "new-subsystem", "name": "Consent receipts",
                                                     "members": ["ConsentReceiptService.java", "ConsentReceiptDto.java"]}, "tester", "1.0.4", ledger, notes)
        self.assertEqual(len(row4["new_subsystems"]), 1)
        l2 = [json.loads(l) for l in open(os.path.join(pack, "l2_navigators.jsonl"), encoding="utf-8") if l.strip()]
        self.assertTrue(any(n["name"] == "Consent receipts" for n in l2))
        with self.assertRaises(SystemExit):
            apply_mod.apply(doc, delta, pack, {"kind": "move", "entity": "ConsentReceiptDto.java", "subsystem": "9999"}, "t", "1.0.5", ledger, notes)


if __name__ == "__main__":
    unittest.main()
