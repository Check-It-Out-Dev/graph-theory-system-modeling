"""Judge and calibration without a model: bank matching, the pointer oracle, batch parsing with a
retry, κ by hand, disputes, anchors and drift, and the events export route."""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "eval", "judge"))

import calibrate  # noqa: E402
import judge  # noqa: E402


class _Proc:
    def __init__(self, stdout):
        self.stdout, self.stderr, self.returncode = stdout, "", 0


def ask_event(rid, q, answer, pointers, terminal="answer", tier="nav-sonnet"):
    return {"event_type": "ask", "request_id": rid, "q": q, "answer": answer, "terminal": terminal, "tier": tier,
            "pointers": [{"name": p} for p in pointers], "user": "haiku-pm", "model": "claude-sonnet-5", "credits": 20.0, "steps": 3}


class RowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank, cls.probes = judge.load_bank()

    def test_bank_matching_by_alias_and_probe(self):
        kind, row = judge.match_question("Purpose of PaymentsDisabledBootGuard", self.bank, self.probes)
        self.assertEqual((kind, row["id"]), ("bank", "BE01"))
        kind, row = judge.match_question("Where is the Kubernetes operator that scales the recommendation engine?", self.bank, self.probes)
        self.assertEqual((kind, row["id"]), ("probe", "OD27"))
        self.assertEqual(judge.match_question("completely unrelated words here zzz", self.bank, self.probes), (None, None))

    def test_pointer_oracle(self):
        be01 = next(r for r in self.bank if r["id"] == "BE01")
        self.assertIn("PaymentsDisabledBootGuard.java", judge.gold_entities(be01))
        has, ok = judge.oracle(ask_event("r", be01["q"], "x", ["PaymentsDisabledBootGuard.java"]), "bank", be01)
        self.assertEqual((has, ok), (True, True))
        has, ok = judge.oracle(ask_event("r", be01["q"], "x", ["Other.java"]), "bank", be01)
        self.assertEqual((has, ok), (True, False))
        content = next(r for r in self.bank if r.get("archetype") == "content" and judge.gold_entities(r))
        self.assertEqual(judge.oracle(ask_event("r", content["q"], "x", list(judge.gold_entities(content))[:1]), "bank", content)[0], False)
        many = ["A.java"] * judge.TOP_POINTERS + ["PaymentsDisabledBootGuard.java"]
        self.assertEqual(judge.oracle(ask_event("r", be01["q"], "x", many), "bank", be01), (True, False))
        od27 = next(p for p in self.probes if p["id"] == "OD27")
        self.assertEqual(judge.oracle(ask_event("r", od27["q"], "", [], terminal="abstain"), "probe", od27), (True, True))
        self.assertEqual(judge.oracle(ask_event("r", od27["q"], "sure", ["X.java"]), "probe", od27), (True, False))

    def test_rows_join_ratings_and_skip_faq(self):
        be01 = next(r for r in self.bank if r["id"] == "BE01")
        events = [ask_event("r1", be01["q"], "answer", ["PaymentsDisabledBootGuard.java"]),
                  ask_event("r2", "What does PaymentsDisabledBootGuard do and when does it stop the app from booting?", "faq", ["PaymentsDisabledBootGuard.java"], tier="faq"),
                  {"event_type": "feedback", "request_id": "r1", "rating": 5, "verified": True, "user": "haiku-pm", "tags": ["great"]}]
        rows = judge.rows_from_events(events, self.bank, self.probes)
        self.assertEqual([r["id"] for r in rows], ["r1"])
        self.assertEqual(rows[0]["human"]["rating"], 5)
        self.assertEqual(rows[0]["qid"], "BE01")
        self.assertTrue(rows[0]["oracle"]["success"])


class JudgeCallTests(unittest.TestCase):
    def test_batches_parse_and_retry_once(self):
        rows = [{"id": f"r{i}", "q": "q", "answer": "a", "pointers": [], "terminal": "answer", "reference": None} for i in range(12)]
        calls = []
        envs = []

        def runner(cmd, env, cwd, timeout):
            calls.append(cmd)
            envs.append(env)
            prompt = cmd[cmd.index("-p") + 1]
            ids = [json.loads(p)["id"] for p in prompt.split("\n\n")[1:]]
            if len(calls) == 1:  # first reply is malformed → retry
                return _Proc(json.dumps({"type": "result", "is_error": False, "result": "```json\n[{bad\n```"}))
            arr = [{"id": i, "located": 5, "grounded": 4, "correct": 5, "abstain": 5, "helpful": 4, "rationale": "fine"} for i in ids]
            return _Proc(json.dumps({"type": "result", "is_error": False, "result": "```json\n" + json.dumps(arr) + "\n```",
                                     "usage": {"input_tokens": 100, "output_tokens": 50}}))
        usage = judge.judge_rows(rows, runner=runner)
        self.assertEqual((usage["calls"], usage["retries"]), (3, 1))
        self.assertTrue(all(r["judge"]["correct"] == 5 for r in rows))
        cmd = calls[0]
        self.assertIn("role=judge", envs[0]["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertNotIn("ANTHROPIC_API_KEY", envs[0])
        self.assertTrue(cmd[cmd.index("--system-prompt-file") + 1].endswith("rubric.md"))

    def test_rate_limit_stops(self):
        rows = [{"id": "r1", "q": "q", "answer": "a", "pointers": [], "terminal": "answer", "reference": None}]
        usage = judge.judge_rows(rows, runner=lambda cmd, env, cwd, timeout: _Proc(json.dumps({"type": "result", "is_error": True, "result": "Rate limit"})))
        self.assertEqual(usage["errors"], 1)
        self.assertIsNone(rows[0].get("judge"))

    def test_parse_scores_tolerates_junk(self):
        self.assertIsNone(judge.parse_scores("nothing"))
        s = judge.parse_scores('x ```json\n[{"id": "a", "located": 4, "grounded": "4", "correct": 3, "abstain": 5, "helpful": 2}, {"nope": 1}]\n```')
        self.assertEqual(s["a"]["grounded"], 4)
        self.assertEqual(len(s), 1)


class CalibrationTests(unittest.TestCase):
    def test_kappa_by_hand(self):
        # 8 agreements of 10, marginals 0.6/0.6 → po 0.8, pe 0.52, κ = 0.5833
        pairs = [(True, True)] * 5 + [(False, False)] * 3 + [(True, False), (False, True)]
        self.assertAlmostEqual(calibrate.cohen_kappa(pairs), 0.5833, places=3)
        self.assertEqual(calibrate.cohen_kappa([(True, True)] * 4), 1.0)
        self.assertIsNone(calibrate.cohen_kappa([(True, True)]))

    def _rows(self):
        rows = []
        for i in range(20):
            good = i % 4 != 0
            rows.append({"id": f"r{i}", "q": "q", "judge": {"located": 5 if good else 2, "correct": 5 if good else 2, "grounded": 4, "abstain": 4, "helpful": 4},
                         "oracle": {"has": True, "success": good if i != 7 else not good},
                         "human": {"rating": 5 if good else 2}, "rr_equiv": 0.9 if good else 0.1, "terminal": "answer"})
        return rows

    def test_calibrate_gate_and_disputes(self):
        rows = self._rows()
        n = judge.flag_disputes(rows)
        self.assertEqual(n, 1)
        res = calibrate.calibrate({"rows": rows}, [], gate=0.6)
        self.assertTrue(res["calibrated"], res)
        self.assertEqual(res["n_oracle"], 20)
        self.assertEqual(res["kappa_human"], 1.0)
        self.assertEqual(res["agreement_rr"], 1.0)
        bad = [dict(r, oracle={"has": True, "success": (i % 2 == 0)}) for i, r in enumerate(rows)]
        res = calibrate.calibrate({"rows": bad}, [], gate=0.6)
        self.assertFalse(res["calibrated"])

    def test_anchor_freeze_and_drift(self):
        import tempfile
        rows = self._rows()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "anchors.jsonl")
            self.assertEqual(calibrate.freeze_anchors(rows, path, 10), 10)
            anchors = calibrate.load_anchors(path)
            same = calibrate.anchor_drift(rows, anchors)
            self.assertEqual((same["matched"], same["mean_abs_delta"], same["drift"]), (10, 0.0, False))
            shifted = [dict(r, judge=dict(r["judge"], correct=max(1, r["judge"]["correct"] - 2))) for r in rows]
            d2 = calibrate.anchor_drift(shifted, anchors)
            self.assertTrue(d2["drift"])


class EventsExportTests(unittest.TestCase):
    def test_admin_events_route(self):
        pack = os.path.join(R, "graph", "pack")
        if not os.path.exists(os.path.join(pack, "entities.csv")):
            self.skipTest("pack absent")
        import tempfile
        from remote import server, telemetry
        d = tempfile.mkdtemp()
        os.environ["CODEMAP_TELEMETRY_DIR"] = d
        for i in range(3):
            telemetry.emit({"event_type": "step", "ts": f"2026-09-1{i}T00:00:00.000Z", "request_id": f"r{i}", "user": "owner", "tier": "engine"}, directory=d)
        app = server.App(token="t", admin_token="adm", navigator=False)
        code, obj = server.handle_events(app, {"Authorization": "Bearer t", "X-CodeMap-Admin": "adm"}, {"since": ["2026-09-11"]})
        self.assertEqual((code, [e["request_id"] for e in obj["events"]]), (200, ["r1", "r2"]))
        self.assertEqual(server.handle_events(app, {"Authorization": "Bearer t"}, {})[0], 401)


class SeedJoinTests(unittest.TestCase):
    def test_a_rephrased_seed_question_joins_through_the_night_file(self):
        bank = [{"id": "BE01", "q": "What does PaymentsDisabledBootGuard do?", "archetype": "locate", "gold_status": "EXECUTED",
                 "gold_answer": "PaymentsDisabledBootGuard.java", "gold_result_excerpt": [["PaymentsDisabledBootGuard.java", "Rule", "11", "x"]]}]
        ev = {"event_type": "ask", "tier": "nav-sonnet", "terminal": "answer", "request_id": "r1", "user": "haiku-pm",
              "q": "hey, which class stops the app booting when payments are switched off?", "answer": "…",
              "pointers": [{"name": "PaymentsDisabledBootGuard.java"}]}
        humans = [{"mode": "codemap", "seed_id": "BE01", "report": {"conversations": [{"turns": [{"request_id": "r1"}]}]}}]
        rows = judge.rows_from_events([ev], bank, [], humans=None)
        self.assertEqual((rows[0]["kind"], rows[0]["oracle"]["has"]), (None, False))
        rows = judge.rows_from_events([ev], bank, [], humans=humans)
        self.assertEqual((rows[0]["kind"], rows[0]["qid"], rows[0]["oracle"]["has"], rows[0]["oracle"]["success"], rows[0]["oracle"]["answered"]), ("bank", "BE01", True, True, True))


if __name__ == "__main__":
    unittest.main()
