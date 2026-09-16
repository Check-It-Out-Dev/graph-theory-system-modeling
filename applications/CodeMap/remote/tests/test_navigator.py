"""Navigator tier without a model: the CLI runner with a fake subprocess, answer parsing, the
engine MCP dispatch in-process, contexts, and the prompt's verb contract."""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "app"))

import claude_cli  # noqa: E402
from remote import contexts, engine_mcp, mcp, navigator  # noqa: E402

PACK = os.path.join(R, "graph", "pack")
HAS_PACK = os.path.exists(os.path.join(PACK, "entities.csv")) and os.path.exists(os.path.join(PACK, "codemap.lbdb"))
PROMPT = os.path.join(R, "prompts", "navigator", "v1.md")


class _Proc:
    def __init__(self, stdout, stderr="", rc=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, rc


def fake_runner(result_obj, capture):
    def runner(cmd, env, cwd, timeout):
        capture.append({"cmd": cmd, "env": env, "cwd": cwd, "timeout": timeout})
        return _Proc(json.dumps(result_obj))
    return runner


RESULT = {"type": "result", "is_error": False, "result": "The guard lives in subsystem 11.\n\n```json\n"
          '{"terminal": "answer", "pointers": ["PaymentsDisabledBootGuard.java"], "confidence": 0.9}\n```',
          "session_id": "sess-1", "num_turns": 3, "duration_ms": 1234, "total_cost_usd": 0.01,
          "usage": {"input_tokens": 12, "output_tokens": 80, "cache_read_input_tokens": 30000, "cache_creation_input_tokens": 0},
          "modelUsage": {"claude-sonnet-5": {"costUSD": 0.01, "provider": "firstParty"}}}


class CliRunnerTests(unittest.TestCase):
    def test_key_is_stripped_and_role_tagged(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-should-not-leak"
        os.environ["CLAUDECODE"] = "1"
        cap = []
        res = claude_cli.run("hi", "claude-haiku-4-5-20251001", role="judge", persona="haiku-pm",
                             runner=fake_runner(RESULT, cap))
        env = cap[0]["env"]
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("CLAUDECODE", env)
        self.assertIn("role=judge", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn("persona=haiku-pm", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertEqual(res["usage"]["cache_read_input_tokens"], 30000)
        self.assertEqual((res["session_id"], res["is_error"], res["num_turns"]), ("sess-1", False, 3))
        cmd = cap[0]["cmd"]
        self.assertIn("--strict-mcp-config", cmd)
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "json")

    def test_cmd_shape_with_mcp_resume_and_system_file(self):
        cmd = claude_cli.build_cmd("q", "claude-sonnet-5", system_file="/p.md", mcp_config={"mcpServers": {}},
                                   allowed_tools=("mcp__engine__*",), max_turns=10, resume="s1", effort="medium", exe="claude")
        self.assertEqual(cmd[cmd.index("--system-prompt-file") + 1], "/p.md")
        self.assertEqual(cmd[cmd.index("--resume") + 1], "s1")
        self.assertEqual(cmd[cmd.index("--allowedTools") + 1], "mcp__engine__*")
        self.assertEqual(cmd[cmd.index("--max-turns") + 1], "10")
        self.assertNotIn("--session-id", cmd)
        cmd2 = claude_cli.build_cmd("q", "m", session_id="new", exe="claude")
        self.assertEqual(cmd2[cmd2.index("--session-id") + 1], "new")

    def test_error_paths_never_raise(self):
        cap = []
        res = claude_cli.run("hi", "m", role="x", runner=fake_runner({"type": "result", "is_error": True,
                                                                      "result": "Rate limit reached, try later"}, cap))
        self.assertTrue(res["is_error"] and res["rate_limited"])
        res = claude_cli.run("hi", "m", role="x", runner=lambda cmd, env, cwd, timeout: _Proc("not json", "boom", 2))
        self.assertTrue(res["is_error"])
        self.assertIn("boom", res["error"])
        self.assertEqual(claude_cli.parse_result("warning line\n{\"a\": 1}"), {"a": 1})


class AnswerParsingTests(unittest.TestCase):
    def test_json_block_is_taken_and_removed(self):
        prose, names, term = navigator.parse_answer(RESULT["result"])
        self.assertEqual((prose, names, term), ("The guard lives in subsystem 11.", ["PaymentsDisabledBootGuard.java"], "answer"))

    def test_abstain_and_fallbacks(self):
        _, _, term = navigator.parse_answer('x\n```json\n{"terminal": "abstain", "pointers": []}\n```')
        self.assertEqual(term, "abstain")
        prose, names, term = navigator.parse_answer("I cannot answer this from the graph.")
        self.assertEqual((names, term), ([], "abstain"))
        prose, names, term = navigator.parse_answer("plain\n```json\n{bad json\n```")
        self.assertEqual(term, "answer")


class ContextTests(unittest.TestCase):
    def test_ownership_and_lru(self):
        c = contexts.Contexts(cap=2)
        a, created = c.get_or_create("a", "owner")
        self.assertTrue(created)
        a2, created = c.get_or_create("a", "owner")
        self.assertIs(a, a2)
        self.assertFalse(created)
        with self.assertRaises(PermissionError):
            c.get_or_create("a", "haiku-pm")
        c.get_or_create("b", "owner")
        c.get_or_create("c", "owner")
        self.assertIsNone(c.get("a"))
        self.assertEqual(len(c), 2)


class PromptContractTests(unittest.TestCase):
    def test_navigator_prompt_declares_exactly_the_navigation_verbs(self):
        sys.path.insert(0, os.path.join(R, "eval", "ci"))
        import harness
        from dsl import VERBS
        text = open(os.path.join(R, "prompts", "navigator", "template.md"), encoding="utf-8").read()
        declared = harness.prompt_verb_table(text)
        nav_verbs = {v: a for v, a in VERBS.items() if v not in ("cache", "answer", "pass")}
        norm = lambda t: {v: tuple(a) if isinstance(a, (list, tuple)) else a for v, a in t.items()}
        self.assertEqual(norm(declared), norm(nav_verbs))

    @unittest.skipUnless(os.path.exists(PROMPT), "v1.md not built")
    def test_built_prompt_is_fresh_and_carries_the_index_and_notes(self):
        text = open(PROMPT, encoding="utf-8").read()
        self.assertNotIn("{{", text)
        self.assertIn("L1 SUBSYSTEM INDEX", text)
        self.assertIn("## Curation notes", text)
        self.assertLess(len(text) // 4, 60000)


@unittest.skipUnless(HAS_PACK, "graph pack absent")
class EngineMcpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.trace = os.path.join(tempfile.gettempdir(), "codemap-test-trace.jsonl")
        if os.path.exists(cls.trace):
            os.remove(cls.trace)
        cls.tools = engine_mcp.EngineTools(trace_file=cls.trace)

    def _call(self, tool, **args):
        _, r = mcp.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                             "params": {"name": tool, "arguments": args}}, self.tools.call,
                            tools=engine_mcp.TOOLS, server_name="codemap-engine")
        return r["result"]

    def test_list_and_step_and_open(self):
        _, r = mcp.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, self.tools.call, tools=engine_mcp.TOOLS)
        self.assertEqual([t["name"] for t in r["result"]["tools"]], ["engine_step", "engine_cypher", "engine_open"])
        res = json.loads(self._call("engine_step", dsl="find(PaymentsDisabledBootGuard)")["content"][0]["text"])
        self.assertEqual(res["kind"], "hits")
        p = json.loads(self._call("engine_open", name="PaymentsDisabledBootGuard.java")["content"][0]["text"])
        self.assertEqual(p["repo"], "backend")
        self.assertTrue(self._call("engine_step", dsl="nonsense()").get("isError"))

    def test_cypher_is_guarded_and_traced(self):
        r = self._call("engine_cypher", stmt="MATCH (e:Entity) WHERE e.curated = 11 RETURN count(*)")
        self.assertNotIn("isError", r)
        r = self._call("engine_cypher", stmt="MATCH (e:Entity) DETACH DELETE e")
        self.assertTrue(r.get("isError"))
        lines = [json.loads(l) for l in open(self.trace, encoding="utf-8")]
        self.assertTrue(any(l["tool"] == "engine_cypher" and l["ok"] is False for l in lines))

    def test_navigator_end_to_end_with_fake_cli(self):
        from remote import server
        sink = []
        app = server.App(token="t", sink=sink, navigator=False)
        nav = navigator.Navigator(app)
        cap = []
        nav.runner = fake_runner(RESULT, cap)
        app.navigator = nav
        code, r = server.handle_mcp(app, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                          "params": {"name": "codemap_ask", "arguments": {"q": "where is the boot guard for payments?"}}},
                                    {"Authorization": "Bearer t", "X-CodeMap-User": "haiku-pm"})
        out = json.loads(r["result"]["content"][0]["text"])
        self.assertEqual((out["tier"], out["terminal"], out["turn"]), ("nav-sonnet", "answer", 1))
        self.assertEqual(out["pointers"][0]["name"], "PaymentsDisabledBootGuard.java")
        self.assertEqual(out["cache_read"], 30000)
        ev = sink[-1]
        self.assertEqual((ev["tier"], ev["session_claude"], ev["tokens"]["cached"]), ("nav-sonnet", "sess-1", 30000))
        # second turn in the same context resumes the Claude session
        server.handle_mcp(app, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                "params": {"name": "codemap_ask", "arguments": {"q": "and what depends on it?", "context_id": out["context_id"]}}},
                          {"Authorization": "Bearer t", "X-CodeMap-User": "haiku-pm"})
        cmd = cap[-1]["cmd"]
        self.assertEqual(cmd[cmd.index("--resume") + 1], "sess-1")
        self.assertTrue(cmd[cmd.index("-p") + 1].startswith("Follow-up"))
        # another user cannot enter the context
        code, r = server.handle_mcp(app, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                          "params": {"name": "codemap_ask", "arguments": {"q": "x", "context_id": out["context_id"]}}},
                                    {"Authorization": "Bearer t", "X-CodeMap-User": "owner"})
        self.assertTrue(r["result"]["isError"])

    def test_budget_exhausted_refuses_before_any_model_call(self):
        from remote import server
        sink = []
        app = server.App(token="t", sink=sink, navigator=False)
        broke = dict(app.users["haiku-pm"], daily_credit_budget=0.0001)
        app.users["haiku-pm"] = broke
        nav = navigator.Navigator(app)
        cap = []
        nav.runner = fake_runner(RESULT, cap)
        app.navigator = nav
        # one paid answer, then the budget is gone
        code, r = server.handle_mcp(app, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                          "params": {"name": "codemap_ask", "arguments": {"q": "where is the boot guard?"}}},
                                    {"Authorization": "Bearer t", "X-CodeMap-User": "haiku-pm"})
        first = json.loads(r["result"]["content"][0]["text"])
        self.assertGreater(first["credits"], 0.0001)
        code, r = server.handle_mcp(app, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                          "params": {"name": "codemap_ask", "arguments": {"q": "and the webhook?"}}},
                                    {"Authorization": "Bearer t", "X-CodeMap-User": "haiku-pm"})
        out = json.loads(r["result"]["content"][0]["text"])
        self.assertEqual((out["terminal"], r["result"].get("isError")), ("budget_exhausted", True))
        self.assertEqual(len(cap), 1, "no second model call")
        self.assertEqual(sink[-1]["event_type"], "budget_refusal")
        self.assertGreater(out["retry_after_s"], 0)
        st = server.handle_budget(app, {"Authorization": "Bearer t"}, {"user": ["haiku-pm"]})
        self.assertEqual(st[0], 429)
        # the engine tools stay free
        code, r = server.handle_mcp(app, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                          "params": {"name": "codemap_step", "arguments": {"dsl": "map()"}}},
                                    {"Authorization": "Bearer t", "X-CodeMap-User": "haiku-pm"})
        self.assertNotIn("isError", r["result"])


if __name__ == "__main__":
    unittest.main()
