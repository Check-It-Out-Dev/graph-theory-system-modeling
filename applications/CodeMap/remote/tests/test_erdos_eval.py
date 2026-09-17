"""The Erdős pairs harness without a model: the phase split of a stream, the two arms' commands (the graph
is the only difference), the assembled prompt, the deterministic checks and the blind order."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "eval", "erdos"))

import erdos_judge as judge  # noqa: E402
import erdos_phases as phases  # noqa: E402
import erdos_prompt as prompt  # noqa: E402
import run_pairs  # noqa: E402


def _start(mid, cache_read=1000):
    return {"type": "stream_event", "event": {"type": "message_start", "message": {"id": mid, "usage": {
        "input_tokens": 1, "cache_creation_input_tokens": 100, "cache_read_input_tokens": cache_read, "output_tokens": 2}}}}


def _delta(output):
    return {"type": "stream_event", "event": {"type": "message_delta", "usage": {
        "input_tokens": 1, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000, "output_tokens": output}}}


def _block(mid, text=None, tool=None):
    block = {"type": "text", "text": text} if text is not None else {"type": "tool_use", "name": tool, "input": {}}
    return {"type": "assistant", "message": {"id": mid, "content": [block],
                                              "usage": {"input_tokens": 1, "cache_creation_input_tokens": 100,
                                                        "cache_read_input_tokens": 1000, "output_tokens": 2}}}


def _result(content):
    return {"type": "user", "message": {"content": [{"type": "tool_result", "content": content}]}}


# one event per content block, the final usage on message_delta: the shape a real stream has
STREAM = [
    (0.0, {"type": "system", "subtype": "init"}),
    (0.9, _start("m1")), (1.0, _block("m1", tool="mcp__graph__graph_query")), (1.1, _delta(10)),
    (1.5, _result("x" * 500)),
    (1.9, _start("m2")), (2.0, _block("m2", tool="Read")), (2.1, _delta(10)),
    (2.5, _result([{"type": "text", "text": "y" * 300}])),
    (2.9, _start("m3")), (3.0, _block("m3", text="## Problem\n" + "a" * 500 + "\n=== ANSWER COMPLETE ===")), (3.0, _delta(10)),
    (3.9, _start("m4")), (4.0, _block("m4", text="Checking one claim.")), (4.2, _block("m4", tool="Grep")), (4.3, _delta(25)),
    (4.5, _result("z" * 50)),
    (5.9, _start("m5")), (6.0, _block("m5", text="=== VERIFIED ===\nno corrections")), (6.0, _delta(10)),
    (6.1, {"type": "result", "subtype": "success", "is_error": False, "num_turns": 5, "usage": {"output_tokens": 65}}),
]


class PhaseTests(unittest.TestCase):
    def test_the_split_counts_each_call_once_and_keeps_the_phases_apart(self):
        s = phases.split(STREAM)
        self.assertTrue(s["marker"])
        self.assertTrue(s["verified_marker"])
        self.assertEqual((s["solve"]["calls"], s["verify"]["calls"], s["total"]["calls"]), (3, 2, 5))
        self.assertEqual(s["solve"]["graph_tool_calls"], 1)
        self.assertEqual(s["solve"]["tools"], {"Read": 1, "mcp__graph__graph_query": 1})
        self.assertEqual(s["solve"]["graph_tool_calls"], 1)
        self.assertEqual(s["verify"]["tools"], {"Grep": 1})
        self.assertEqual(s["solve"]["result_bytes"], 800)
        self.assertEqual(s["verify"]["result_bytes"], 50)
        self.assertEqual(s["verify"]["tokens"]["output_tokens"], 25 + 10)      # the final usage of m4 and m5
        self.assertEqual(s["solve"]["tokens_sum"], 3 * 1111)
        self.assertEqual(s["solve"]["tokens_weighted"], round(3 * (1 + 200 + 100 + 50), 1))
        self.assertEqual(s["result"]["usage"]["output_tokens"], s["total"]["tokens"]["output_tokens"])
        self.assertEqual((s["solve"]["seconds"], s["verify"]["seconds"]), (3.0, 3.1))
        self.assertTrue(s["answer"].startswith("## Problem"))
        self.assertEqual(s["corrections"], "no corrections")

    def test_a_recorded_stream_sums_to_its_result(self):
        events = phases.load(os.path.join(R, "eval", "erdos", "fixtures", "stream.haiku.events.jsonl"))
        s = phases.split(events)
        result = s["result"]["usage"]
        self.assertEqual(s["total"]["tokens"], result)                      # every key, per-call sums = run totals
        self.assertEqual(s["total"]["calls"], 2)
        self.assertEqual(s["total"]["tools"], {"Glob": 1})

    def test_a_run_without_the_marker_is_all_solve(self):
        s = phases.split(STREAM[:9])
        self.assertFalse(s["marker"])
        self.assertEqual((s["solve"]["calls"], s["verify"]["calls"]), (2, 0))

    def test_events_file_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.events.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for t, e in STREAM:
                    f.write(json.dumps({"t": t, "event": e}) + "\n")
            self.assertEqual(phases.split(phases.load(p))["total"]["calls"], 5)
            self.assertTrue(run_pairs.finished(p))


class ArmTests(unittest.TestCase):
    def test_the_graph_is_the_only_difference(self):
        problem = {"id": "p", "prompt": "Design X."}
        files = {"mcp": "graph.json", "context": "ctx"}
        g = run_pairs.command("general", problem, "claude-opus-5", 100, files, exe="claude")
        e = run_pairs.command("erdos", problem, "claude-opus-5", 100, files, exe="claude")
        for cmd in (g, e):
            self.assertEqual(cmd[1:3], ["-p", run_pairs.TASK_CARD.format(problem="Design X.")])
            for flag in ("--restricted", "--strict-mcp-config", "--no-session-persistence"):
                self.assertIn(flag, cmd)
            self.assertEqual(cmd[cmd.index("--tools") + 1], "Read,Grep,Glob")
            self.assertEqual(cmd[cmd.index("--model") + 1], "claude-opus-5")
        self.assertNotIn("--mcp-config", g)
        self.assertNotIn("--add-dir", g)
        self.assertEqual(e[e.index("--mcp-config") + 1], "graph.json")
        self.assertEqual(e[e.index("--add-dir") + 1], "ctx")
        self.assertIn("mcp__graph__*", e[e.index("--allowedTools") + 1])
        self.assertNotIn("mcp__graph__*", g[g.index("--allowedTools") + 1])
        extra = [x for x in e if x not in g]
        self.assertEqual(sorted(extra), sorted(["--mcp-config", "graph.json", "Read,Grep,Glob,mcp__graph__*", "--add-dir", "ctx"]))
        self.assertNotIn("--append-system-prompt-file", e)
        self.assertEqual(run_pairs.RUN_ENV, {"CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD": "1"})

    def test_the_task_card_has_one_phase(self):
        self.assertIn(phases.SOLVED, run_pairs.TASK_CARD)
        self.assertNotIn(phases.VERIFIED, run_pairs.TASK_CARD)            # no verification round (owner, 2026-09-17)
        self.assertNotIn("Phase 2", run_pairs.TASK_CARD)

    def test_the_preflight_passes_only_on_the_checksum(self):
        files = {"context": "ctx"}
        cmd = run_pairs.preflight_command(files, exe="claude")
        for flag in ("--restricted", "--strict-mcp-config", "--add-dir"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--add-dir") + 1], "ctx")

        def fake(said, is_error=False):
            def run(cmd, **kw):
                self.assertEqual(kw["env"]["CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD"], "1")
                for name in run_pairs.BLOCKING_ENV + ("ANTHROPIC_API_KEY",):
                    self.assertNotIn(name, kw["env"])
                return subprocess.CompletedProcess(cmd, 0, json.dumps({"result": said, "is_error": is_error,
                                                                       "usage": {"input_tokens": 3, "cache_read_input_tokens": 20000}}), "")
            return run

        os.environ["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
        try:
            self.assertTrue(run_pairs.preflight(files, "0123456789abcdef", ".", run=fake("0123456789abcdef"))[0])
            self.assertFalse(run_pairs.preflight(files, "0123456789abcdef", ".", run=fake("NONE"))[0])
            self.assertFalse(run_pairs.preflight(files, "0123456789abcdef", ".", run=fake("0123456789abcdef", True))[0])
            self.assertFalse(run_pairs.preflight(files, None, ".", run=fake("None"))[0])
        finally:
            del os.environ["CLAUDE_CODE_DISABLE_CLAUDE_MDS"]

    def test_the_context_directory_holds_only_claude_md(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = run_pairs.context_dir(d, "<erdos/>")
            self.assertEqual(os.listdir(ctx), ["CLAUDE.md"])

    def test_the_graph_config_uses_absolute_paths(self):
        cfg = run_pairs.graph_config("graph/pack")["mcpServers"]["graph"]
        self.assertTrue(cfg["args"][0].endswith("ladybug_mcp.py"))
        self.assertTrue(os.path.isabs(cfg["args"][0]))
        self.assertTrue(os.path.isabs(cfg["env"]["CODEMAP_PACK_DIR"]))


class PromptTests(unittest.TestCase):
    def test_a_candidate_body_renders_with_the_checksum_last(self):
        text = prompt.assemble(body="<erdos_manual>\n<!-- note -->\nCANDIDATE\n</erdos_manual>")
        self.assertTrue(text.startswith("<erdos_manual>\nCANDIDATE\n</erdos_manual>\n<manual_checksum value="))
        above = text[:text.index("<manual_checksum")]
        self.assertEqual(prompt.checksum(text), prompt.hashlib.sha256(above.encode("utf-8")).hexdigest()[:16])
        self.assertIsNone(prompt.checksum(above))
        self.assertTrue(prompt.version(text).startswith("erdos@"))

    def test_the_committed_manual_is_the_rendered_skill(self):
        if not os.path.exists(os.path.join(R, "graph", "pack", "codemap.lbdb")):
            self.skipTest("no pack")
        with open(prompt.MANUAL, encoding="utf-8") as f:
            committed = f.read().replace("\r\n", "\n")
        self.assertEqual(committed, prompt.assemble())
        self.assertFalse(prompt.skill_body().startswith("---"))
        self.assertIsNotNone(prompt.checksum(committed))


class JudgeTests(unittest.TestCase):
    def test_recall_unknown_files_and_blind_order(self):
        answer = "Change `PaymentsDisabledBootGuard.java` and plan-billing.component.ts; also Invented.java."
        must = [{"name": "PaymentsDisabledBootGuard.java"}, {"name": "StripeWebhookHandler.java"}]
        self.assertEqual(judge.recall(answer, must), {"hits": ["PaymentsDisabledBootGuard.java"], "n": 2, "recall": 0.5})
        idx = {"paymentsdisabledbootguard.java", "plan-billing.component.ts"}
        self.assertEqual(judge.unknown_files(answer, idx), {"mentioned": 3, "unknown": ["invented.java"]})
        self.assertEqual(judge.blind_order("x"), judge.blind_order("x"))
        self.assertEqual(sorted(judge.blind_order("x")), ["erdos", "general"])
        prompt_text = judge.judge_prompt({"prompt": "P", "gold": {"must_find": must, "key_facts": []}}, "AAA", "BBB")
        self.assertLess(prompt_text.index("ANSWER A"), prompt_text.index("ANSWER B"))
        self.assertNotIn("erdos", prompt_text.lower())
        self.assertNotIn("general", prompt_text.lower())


if __name__ == "__main__":
    unittest.main()
