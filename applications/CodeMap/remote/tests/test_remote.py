"""Remote MCP — contract tests without sockets. Run: python -m unittest discover -s applications/CodeMap/remote/tests

The engine-backed checks need the pack (gitignored data; fetched from the Release in CI by S5) and
skip with a reason when it is absent; the protocol, auth, user-enum and telemetry checks never skip.
"""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)

from remote import ids, mcp, pointers, telemetry, users  # noqa: E402

PACK = os.path.join(R, "graph", "pack")
HAS_PACK = os.path.exists(os.path.join(PACK, "entities.csv")) and os.path.exists(os.path.join(PACK, "codemap.lbdb"))


def _rpc(method, mid=1, **params):
    m = {"jsonrpc": "2.0", "id": mid, "method": method}
    if params:
        m["params"] = params
    return m


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def call_tool(name, args):
            self.calls.append((name, args))
            return json.dumps({"echo": name}), False
        self.call_tool = call_tool

    def test_initialize_echoes_a_known_version_and_defaults_otherwise(self):
        code, r = mcp.dispatch(_rpc("initialize", protocolVersion="2025-06-18"), self.call_tool)
        self.assertEqual((code, r["result"]["protocolVersion"]), (200, "2025-06-18"))
        code, r = mcp.dispatch(_rpc("initialize", protocolVersion="1999-01-01"), self.call_tool)
        self.assertEqual(r["result"]["protocolVersion"], mcp.PROTOCOL_VERSIONS[0])
        self.assertIn("tools", r["result"]["capabilities"])

    def test_tools_list_names_the_seven_tools(self):
        _, r = mcp.dispatch(_rpc("tools/list"), self.call_tool)
        self.assertEqual([t["name"] for t in r["result"]["tools"]],
                         ["codemap_ask", "codemap_step", "codemap_open", "codemap_search",
                          "codemap_feedback", "codemap_miss", "codemap_status"])
        for t in r["result"]["tools"]:
            self.assertEqual(t["inputSchema"]["type"], "object")

    def test_notification_is_202_without_body(self):
        self.assertEqual(mcp.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}, self.call_tool), (202, None))

    def test_unknown_method_and_unknown_tool(self):
        _, r = mcp.dispatch(_rpc("resources/list"), self.call_tool)
        self.assertEqual(r["error"]["code"], -32601)
        _, r = mcp.dispatch(_rpc("tools/call", name="nope", arguments={}), self.call_tool)
        self.assertEqual(r["error"]["code"], -32602)

    def test_tool_call_wraps_text_and_marks_errors(self):
        _, r = mcp.dispatch(_rpc("tools/call", name="codemap_status", arguments={}), self.call_tool)
        self.assertEqual(r["result"]["content"][0]["type"], "text")
        self.assertNotIn("isError", r["result"])
        self.assertEqual(self.calls, [("codemap_status", {})])

        def boom(name, args):
            raise RuntimeError("engine down")
        _, r = mcp.dispatch(_rpc("tools/call", name="codemap_step", arguments={"dsl": "map()"}), boom)
        self.assertTrue(r["result"]["isError"])
        self.assertIn("engine down", r["result"]["content"][0]["text"])

    def test_batch(self):
        code, r = mcp.dispatch_body([_rpc("ping", 1), {"jsonrpc": "2.0", "method": "notifications/x"}, _rpc("ping", 2)], self.call_tool)
        self.assertEqual((code, [x["id"] for x in r]), (200, [1, 2]))

    def test_invalid_request(self):
        code, r = mcp.dispatch({"id": 1}, self.call_tool)
        self.assertEqual((code, r["error"]["code"]), (400, -32600))


class UserEnumTests(unittest.TestCase):
    def setUp(self):
        self.users = users.load()

    def test_enum_ids_are_label_safe_and_budgeted(self):
        for uid, u in self.users.items():
            self.assertRegex(uid, users.ID_RX.pattern)
            self.assertGreater(u["daily_credit_budget"], 0)
        self.assertIn("anonymous", self.users)
        self.assertEqual(len([u for u in self.users.values() if u["kind"] == "persona"]), 6)

    def test_resolution_order_body_header_default(self):
        u, w = users.resolve(self.users, {"X-CodeMap-User": "haiku-pm"}, {"user": "owner"})
        self.assertEqual((u["id"], w), ("owner", "owner"))
        u, w = users.resolve(self.users, {"X-CodeMap-User": "haiku-pm"}, {})
        self.assertEqual(u["id"], "haiku-pm")
        u, w = users.resolve(self.users, {}, None)
        self.assertEqual(u["id"], "anonymous")
        u, w = users.resolve(self.users, {"X-CodeMap-User": "nobody"}, None)
        self.assertEqual((u, w), (None, "nobody"))


class TelemetryTests(unittest.TestCase):
    def test_emit_validates_and_sorts(self):
        sink = []
        ev = telemetry.emit({"event_type": "ask", "ts": telemetry.now_iso(), "request_id": "r1",
                             "user": "owner", "tier": "faq", "credits": 0.0}, sink=sink)
        self.assertEqual(ev["schema"], 1)
        self.assertEqual(sink, [ev])
        with self.assertRaises(ValueError):
            telemetry.emit({"event_type": "ask", "ts": "t", "request_id": "r", "user": "o", "tier": "gold"}, sink=[])
        with self.assertRaises(ValueError):
            telemetry.emit({"event_type": "ask", "ts": "t", "request_id": "r", "user": "o", "tier": "faq", "bogus": 1}, sink=[])
        with self.assertRaises(ValueError):
            telemetry.emit({"event_type": "ask", "ts": "t", "request_id": "r", "user": "o", "tier": "faq", "steps": "3"}, sink=[])

    def test_meter_maps_anthropic_usage(self):
        m = telemetry.Meter().add({"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100,
                                   "cache_creation_input_tokens": 7})
        m.add({"input_tokens": 1, "output_tokens": 1})
        self.assertEqual(m.snapshot(), {"prompt": 11, "completion": 6, "cached": 100, "cache_creation": 7})
        self.assertEqual(m.calls, 2)

    def test_events_round_trip_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            telemetry.emit({"event_type": "step", "ts": telemetry.now_iso(), "request_id": "r1",
                            "user": "owner", "tier": "engine"}, directory=d)
            telemetry.emit({"event_type": "step", "ts": telemetry.now_iso(), "request_id": "r2",
                            "user": "owner", "tier": "engine"}, directory=d)
            self.assertEqual([e["request_id"] for e in telemetry.iter_events(d)], ["r1", "r2"])


class IdsAndPointersTests(unittest.TestCase):
    def test_prompt_version_is_stable_and_absent_is_named(self):
        self.assertEqual(ids.prompt_version("/nope/none.md"), "nav@none")
        self.assertEqual(ids.sha16("abc"), ids.sha16(b"abc"))
        self.assertEqual(len(ids.sha16("abc")), 16)

    def test_pointer_strips_the_authoring_path(self):
        p = pointers.to_pointer({"name": "X.java", "file_path": "C:/Users/n/IdeaProjects/checkItOut-be2/src/main/X.java",
                                 "entity_type": "Rule", "curated": "11", "line_count": "40", "entry_point": "True"})
        self.assertEqual((p["repo"], p["path"], p["subsystem"], p["lines"], p["entry_point"]),
                         ("backend", "src/main/X.java", 11, 40, True))
        self.assertNotIn("Users", json.dumps(p))
        self.assertEqual(pointers.split_path("/x/y/z.ts"), ("?", "z.ts"))


@unittest.skipUnless(HAS_PACK, "graph pack absent (fetch the Release asset)")
class ServerWithPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from remote import server
        cls.server = server
        cls.sink = []
        cls.app = server.App(token="t0k", admin_token="adm", sink=cls.sink, navigator=False)  # never the model in tests

    def _post(self, body, **headers):
        h = {"Authorization": "Bearer t0k", "X-CodeMap-User": "owner"}
        h.update(headers)
        return self.server.handle_mcp(self.app, body, h)

    def test_auth_matrix(self):
        self.assertEqual(self.server.handle_mcp(self.app, _rpc("ping"), {})[0], 401)
        self.assertEqual(self.server.handle_mcp(self.app, _rpc("ping"), {"Authorization": "Bearer wrong"})[0], 401)
        code, r = self._post(_rpc("ping"), **{"X-CodeMap-User": "nobody"})
        self.assertEqual((code, r["error"]), (400, "unknown_user"))
        self.assertEqual(self._post(_rpc("ping"))[0], 200)
        self.assertEqual(self.server.handle_status(self.app, {})[0], 401)
        self.assertEqual(self.server.handle_healthz(self.app)[0], 200)

    def test_step_and_open_return_pointers_not_paths(self):
        _, r = self._post(_rpc("tools/call", name="codemap_step", arguments={"dsl": "read(PaymentsDisabledBootGuard.java)"}))
        res = json.loads(r["result"]["content"][0]["text"])
        self.assertEqual(res["pointer"]["repo"], "backend")
        self.assertNotIn("Users/", json.dumps(res))
        _, r = self._post(_rpc("tools/call", name="codemap_open", arguments={"name": "PaymentsDisabledBootGuard.java"}))
        p = json.loads(r["result"]["content"][0]["text"])
        self.assertEqual(p["subsystem"], 11)
        self.assertIn("clue", p)
        _, r = self._post(_rpc("tools/call", name="codemap_open", arguments={"name": "Nope.java"}))
        self.assertTrue(r["result"]["isError"])

    def test_bad_dsl_is_a_tool_error_not_a_crash(self):
        _, r = self._post(_rpc("tools/call", name="codemap_step", arguments={"dsl": "explode(everything)"}))
        self.assertTrue(r["result"]["isError"])

    def test_ask_faq_hit_writes_one_event_with_pointers(self):
        n = len(self.sink)
        q = "What does PaymentsDisabledBootGuard do and when does it stop the app from booting?"
        _, r = self._post(_rpc("tools/call", name="codemap_ask", arguments={"q": q}))
        out = json.loads(r["result"]["content"][0]["text"])
        self.assertEqual((out["tier"], out["credits"]), ("faq", 0.0))
        self.assertTrue(out["pointers"] and out["pointers"][0]["repo"] == "backend")
        self.assertEqual(len(self.sink), n + 1)
        ev = self.sink[-1]
        self.assertEqual((ev["event_type"], ev["tier"], ev["faq_cache"], ev["user"]), ("ask", "faq", "hit", "owner"))
        self.assertTrue(ev["pack_version"] and ev["prompt_version"])

    def test_ask_miss_without_navigator_descends(self):
        _, r = self._post(_rpc("tools/call", name="codemap_ask", arguments={"q": "zzz nothing like this exists qqq"}))
        out = json.loads(r["result"]["content"][0]["text"])
        self.assertEqual(out["terminal"], "descend")
        self.assertIn("l1_index", out)

    def test_status_and_budget(self):
        _, r = self._post(_rpc("tools/call", name="codemap_status", arguments={}))
        st = json.loads(r["result"]["content"][0]["text"])
        self.assertEqual(st["you"]["user"], "owner")
        code, b = self.server.handle_budget(self.app, {"Authorization": "Bearer t0k"}, {"user": ["haiku-pm"]})
        self.assertEqual((code, b["budget"]), (200, 200))
        self.assertEqual(self.server.handle_budget(self.app, {"Authorization": "Bearer t0k"}, {"user": ["ghost"]})[0], 400)


if __name__ == "__main__":
    unittest.main()
