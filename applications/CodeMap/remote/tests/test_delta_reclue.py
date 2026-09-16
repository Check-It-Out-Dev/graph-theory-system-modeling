"""Reclue without a model: the dossier is built from the pack, the gates refuse derived numbers and
invented files, a fake CLI's prose lands with provenance and a snapshot, untouched lines stay
byte-identical, and a failed model call is recorded rather than raised."""

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "graph", "delta"))

import reclue  # noqa: E402

PACK = os.path.join(R, "graph", "pack")
HAS_PACK = os.path.exists(os.path.join(PACK, "entities.csv"))


def _fake_runner(prose):
    class P:
        def __init__(self, o):
            self.stdout, self.stderr, self.returncode = json.dumps(o), "", 0
    return lambda cmd, env, cwd, timeout: P({"type": "result", "is_error": False, "result": json.dumps(prose),
                                              "structured_output": prose, "usage": {"input_tokens": 10, "output_tokens": 5},
                                              "session_id": "s", "num_turns": 1})


class GateTests(unittest.TestCase):
    doss = {"sub_id": "6", "name": "Two-factor auth", "size": 51, "layer_profile": {"Resource": 27},
            "entry_points": ["UserCacheService.java (51 ext in-edges)"], "spines": [], "contracts": [], "seams": [],
            "members": [{"name": "UserCacheService.java", "type": "Process", "ext_in": 51, "ext_out": 0},
                        {"name": "UserServiceUnitTest.java", "type": "Rule", "ext_in": 0, "ext_out": 3}], "curation_notes": [],
            "current_prose": {}}

    def test_pass(self):
        ok = {"ai_summary": "Second factor and the user cache; UserCacheService.java carries 51 external in-edges.",
              "responsibilities": ["TOTP flows", "user cache", "step-up tokens"], "caveats": []}
        self.assertEqual(reclue.gates(ok, self.doss), [])

    def test_refuses_derived_numbers_and_invented_files(self):
        bad = {"ai_summary": "About 60 files; GhostService.java is the root.", "responsibilities": ["a", "b"], "caveats": None}
        probs = reclue.gates(bad, self.doss)
        self.assertTrue(any("number not in the dossier: 60" in p for p in probs))
        self.assertTrue(any("file not in the dossier: GhostService.java" in p for p in probs))
        self.assertTrue(any("responsibilities 2" in p for p in probs))
        self.assertTrue(any("caveats" in p for p in probs))
        suffix = {"ai_summary": "Every *UnitTest.java here mocks UserCacheService.java; 3 seams, ratio 0.27.", "responsibilities": ["a", "b", "c"], "caveats": []}
        probs = reclue.gates(suffix, self.doss)
        self.assertFalse(any("UnitTest.java" in p for p in probs), probs)  # a suffix pattern of a dossier file
        self.assertFalse(any("dossier: 3" in p for p in probs), probs)  # a small count
        self.assertTrue(any("dossier: 0.27" in p for p in probs), probs)  # a derived ratio is not copied
        neighbour = {"ai_summary": "Calls TotpEncryptionService.java next door.", "responsibilities": ["a", "b", "c"], "caveats": []}
        self.assertTrue(any("TotpEncryptionService.java" in p for p in reclue.gates(neighbour, self.doss)))
        self.assertEqual([p for p in reclue.gates(neighbour, self.doss, known={"TotpEncryptionService.java"}) if "file" in p], [])
        long = {"ai_summary": " ".join(["word"] * 81), "responsibilities": ["a", "b", "c"], "caveats": []}
        self.assertTrue(any("> 80" in p for p in reclue.gates(long, self.doss)))


@unittest.skipUnless(HAS_PACK, "pack absent")
class ReclueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="codemap-reclue-")
        self.pack = os.path.join(self.tmp, "pack")
        shutil.copytree(PACK, self.pack, ignore=shutil.ignore_patterns("codemap.lbdb"))
        self.ledger = os.path.join(self.tmp, "ledger")

    def test_dossier_from_the_pack(self):
        ents, edges, lines = reclue.load_pack(self.pack)
        l2 = [json.loads(l) for l in lines if l.strip()]
        d = reclue.dossier(self.pack, 6, ents, edges, l2)
        self.assertEqual(d["name"], "Two-factor auth & user cache")
        self.assertGreater(d["size"], 40)
        self.assertEqual(d["members"][0]["name"], "UserCacheService.java")  # highest external in-degree first
        self.assertTrue(any(s["name"] for s in d["seams"]))
        self.assertIn("ai_summary", d["current_prose"])

    def test_reclue_writes_only_the_touched_line(self):
        before = open(os.path.join(self.pack, "l2_navigators.jsonl"), encoding="utf-8").read().split("\n")
        ents, edges, lines = reclue.load_pack(self.pack)
        top = reclue.dossier(self.pack, 6, ents, edges, [json.loads(l) for l in lines if l.strip()])["members"][0]
        prose = {"ai_summary": f"TOTP second factor and the Firestore-backed user cache; {top['name']} is the entry with {top['ext_in']} external in-edges.",
                 "responsibilities": ["TOTP + step-up flows", "UserCacheService.java / FirestoreService.java", "step-up token plumbing"],
                 "caveats": ["UserCacheService.java single-carries the seam to subsystem 11"]}
        rep = reclue.reclue(self.pack, ["6"], "1.0.9", self.ledger, backend="claude", runner=_fake_runner(prose))
        self.assertEqual(rep["reclued"], ["6"])
        after = open(os.path.join(self.pack, "l2_navigators.jsonl"), encoding="utf-8").read().split("\n")
        self.assertEqual(len(before), len(after))
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        self.assertEqual(len(changed), 1)
        nav = json.loads(after[changed[0]])
        self.assertEqual((str(nav["sub_id"]), nav["clue_version"], nav["clue_body_status"], nav["generated_by"]),
                         ("6", "delta-1.0.9", "CURRENT", "ErdosNavigator/reclue-delta"))
        self.assertEqual(nav["ai_summary"], prose["ai_summary"])
        snap = json.load(open(os.path.join(self.ledger, "1.0.9.reclue.json"), encoding="utf-8"))
        self.assertEqual(snap["results"][0]["status"], "reclued")
        self.assertIn("ai_summary", snap["results"][0]["before"])
        self.assertEqual(snap["results"][0]["usage"]["input_tokens"], 10)

    def test_gated_prose_and_model_errors_are_recorded_not_written(self):
        before = open(os.path.join(self.pack, "l2_navigators.jsonl"), encoding="utf-8").read()
        bad = {"ai_summary": "Roughly 999 files.", "responsibilities": ["a", "b", "c"], "caveats": []}
        rep = reclue.reclue(self.pack, ["6"], "1.0.9", self.ledger, backend="claude", runner=_fake_runner(bad))
        self.assertEqual((rep["reclued"], rep["results"][0]["status"]), ([], "gated"))

        class P:
            stdout, stderr, returncode = json.dumps({"type": "result", "is_error": True, "result": "Rate limit"}), "", 0
        rep = reclue.reclue(self.pack, ["6", "10"], "1.0.9", self.ledger, backend="claude", runner=lambda *a, **k: P())
        self.assertEqual([r["status"] for r in rep["results"]], ["skipped", "skipped"])
        self.assertTrue(rep["results"][0]["problems"][0].startswith("model:"))
        self.assertEqual(open(os.path.join(self.pack, "l2_navigators.jsonl"), encoding="utf-8").read(), before)
        rep = reclue.reclue(self.pack, ["6"], "1.0.9", self.ledger, backend="none")
        self.assertEqual(rep["results"][0]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
