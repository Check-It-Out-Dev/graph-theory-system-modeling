"""The prompt-under-test runner without a model or Maven: the session's command line, the output-token budget that
ends a session, the stream file, the trace the checks and the judge read, the rendered prompt and its checksum."""

import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "eval", "put"))

import put_contract  # noqa: E402
import put_prompt  # noqa: E402
import put_runner  # noqa: E402
import put_tasks  # noqa: E402

INSTANCE = "backend-conventions"


def _stream_event(kind, **kw):
    return {"type": "stream_event", "event": dict(type=kind, **kw)}


def _session(messages, extra=()):
    """A stream-json session: init, then per message a start and a delta with its output tokens."""
    lines = [{"type": "system", "subtype": "init", "model": "claude-sonnet-5", "tools": ["Read"]}]
    for i, out in enumerate(messages):
        lines.append(_stream_event("message_start", message={"id": f"msg{i}", "usage": {"output_tokens": 1}}))
        lines.append(_stream_event("content_block_delta", delta={"text": "x"}))
        lines.append(_stream_event("message_delta", usage={"output_tokens": out}))
    lines.extend(extra)
    lines.append({"type": "result", "subtype": "success", "num_turns": len(messages), "is_error": False,
                  "usage": {"input_tokens": 10, "output_tokens": sum(messages)}, "result": "done"})
    return "\n".join(json.dumps(x) for x in lines) + "\n"


class FakeProc:
    def __init__(self, text):
        self.stdout = io.StringIO(text)
        self.stderr = io.StringIO("")
        self.returncode = None
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.contract = put_contract.load(INSTANCE)
        self.tmp = tempfile.mkdtemp()

    def _run(self, text, budget):
        procs = []

        def popen(*a, **kw):
            procs.append(FakeProc(text))
            return procs[-1]

        path = os.path.join(self.tmp, "events.jsonl")
        res = put_runner.run_session(["claude"], self.tmp, path, 60, budget, {}, popen=popen)
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        return res, rows, procs[0]

    def test_the_command_is_restricted_with_a_narrow_bash_allowlist(self):
        cmd = put_runner.command("CARD", {"mcp": "m.json", "context": "ctx"}, self.contract["runner"], "sonnet", exe="claude")
        self.assertEqual(cmd[:4], ["claude", "-p", "CARD", "--model"])
        self.assertIn("--restricted", cmd)
        self.assertNotIn("--safe-mode", cmd)
        allowed = cmd[cmd.index("--allowedTools") + 1].split(",")
        self.assertNotIn("Bash", allowed)                                      # never bare Bash (S0: Bash escapes the worktree)
        self.assertIn("Bash(./mvnw *)", allowed)
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "dontAsk")
        self.assertEqual(cmd[cmd.index("--add-dir") + 1], "ctx")
        self.assertEqual(cmd[cmd.index("--max-turns") + 1], "150")

    def test_a_session_under_budget_runs_to_its_result(self):
        res, rows, proc = self._run(_session([100, 200]), budget=1000)
        self.assertFalse(res["budget_exhausted"])
        self.assertFalse(proc.killed)
        self.assertEqual(res["output_tokens"], 300)
        kinds = [r["event"].get("type") for r in rows]
        self.assertIn("result", kinds)
        self.assertNotIn("content_block_delta", [r["event"].get("event", {}).get("type") for r in rows])   # only start/delta kept

    def test_a_session_over_budget_is_ended_and_says_so(self):
        res, rows, proc = self._run(_session([600, 600, 600]), budget=1000)
        self.assertTrue(res["budget_exhausted"])
        self.assertTrue(proc.killed)
        self.assertEqual(rows[-1]["event"]["type"], "budget_exhausted")
        self.assertGreater(rows[-1]["event"]["output_tokens"], 1000)
        self.assertNotIn("result", [r["event"].get("type") for r in rows])

    def test_the_trace_lists_tool_calls_in_order_with_results(self):
        events = [(0, {"type": "assistant", "message": {"content": [
                      {"type": "tool_use", "id": "a", "name": "mcp__graph__graph_query", "input": {"statement": "MATCH (e) RETURN e"}},
                      {"type": "tool_use", "id": "b", "name": "Read", "input": {"file_path": "src/main/X.java"}}]}}),
                  (1, {"type": "user", "message": {"content": [
                      {"type": "tool_result", "tool_use_id": "a", "content": "rows"},
                      {"type": "tool_result", "tool_use_id": "b", "content": [{"type": "text", "text": "class X"}], "is_error": True}]}})]
        text = put_runner.trace_md(events)
        self.assertEqual(text.splitlines()[0], "1. mcp__graph__graph_query MATCH (e) RETURN e -> rows")
        self.assertEqual(text.splitlines()[1], "2. Read src/main/X.java [error] -> class X")

    def test_the_child_never_sees_an_api_key_and_carries_its_role(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        try:
            env = put_runner.child_env("coder", "put@abc", "lbl")
        finally:
            del os.environ["ANTHROPIC_API_KEY"]
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertEqual(env["CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD"], "1")
        self.assertIn("role=coder", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn("prompt_version=put@abc", env["OTEL_RESOURCE_ATTRIBUTES"])

    def test_the_rendered_prompt_expands_its_includes_and_ends_with_its_checksum(self):
        body = put_prompt.load_body(INSTANCE)
        text = put_prompt.render(body, INSTANCE)
        self.assertNotIn("<include file=", text)
        self.assertNotIn("<!--", text)
        self.assertIn("mcp__graph__graph_query", text)            # the tool contract arrived from references/tools.md
        self.assertEqual(put_prompt.checksum(text), put_prompt.sha16(text[:text.rindex("<manual_checksum")]))
        self.assertTrue(put_prompt.version(body).startswith("put@"))

    def test_the_card_states_the_task_and_the_interface_and_no_convention(self):
        for task in put_contract.tasks(INSTANCE):
            card = put_tasks.card(task)
            self.assertIn(task["text"], card)
            for word in ("@SchedulerLock", "AFTER_COMMIT", "Liquibase", "@Version", "@PreAuthorize", "graph",
                         "ANSWER COMPLETE", "TranslatableException", "DtoIn", "UnitTest"):
                if word in ("DtoIn",) and "DtoIn" in " ".join(task["interface"]):
                    continue                                            # an interface may name its own DTO type
                self.assertNotIn(word, card, f"{task['id']}: the card gives away {word}")


if __name__ == "__main__":
    unittest.main()
