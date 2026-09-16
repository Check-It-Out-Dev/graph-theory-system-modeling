"""The night planner and runner without a model: deterministic plans, non-empty slices, prompt and
MCP-config shapes, the report parser, a rate-limited night that stops as partial, agent sync."""

import json
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "eval", "humans"))

import run_night  # noqa: E402


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = run_night.load_personas()
        cls.bank, cls.probes = run_night.load_bank()

    def test_every_persona_has_a_slice_and_a_role_file(self):
        for p in self.cfg["personas"]:
            self.assertGreater(len(run_night.slice_for(p, self.bank, self.probes)), 5, p["id"])
            self.assertTrue(os.path.exists(os.path.join(R, "eval", "humans", p["role_file"])))
        self.assertEqual(len(self.probes), 30)
        self.assertEqual({p["expect"] for p in self.probes}, {"answer", "abstain"})

    def test_plan_is_deterministic_and_ordered_haiku_first(self):
        a = run_night.plan("2026-09-17", self.cfg, self.bank, self.probes, seed=1)
        b = run_night.plan("2026-09-17", self.cfg, self.bank, self.probes, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(a[0]["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(a[-1]["model"], "claude-opus-5")
        partners = sum(1 for x in a if x.get("paired_with"))
        self.assertEqual(len(a) - partners, sum(p["conversations_per_night"] for p in self.cfg["personas"]))
        c = run_night.plan("2026-09-18", self.cfg, self.bank, self.probes, seed=1)
        self.assertNotEqual([x["seed_id"] for x in a], [x["seed_id"] for x in c])

    def test_filters(self):
        only = run_night.plan("2026-09-17", self.cfg, self.bank, self.probes, only="haiku-pm", limit=1)
        self.assertEqual([(x["persona"], x["mode"]) for x in only], [("haiku-pm", "codemap")])
        haiku = run_night.plan("2026-09-17", self.cfg, self.bank, self.probes, haiku_only=True)
        self.assertTrue(all(x["model"].startswith("claude-haiku") for x in haiku))

    def test_baseline_share_over_many_nights(self):
        n = base = 0
        for d in range(30):
            for c in run_night.plan(f"2026-10-{d+1:02d}", self.cfg, self.bank, self.probes):
                n += 1
                base += c["mode"] == "baseline"
        self.assertTrue(0.08 < base / n < 0.3, base / n)

    def test_prompt_and_mcp_shapes(self):
        c = run_night.plan("2026-09-17", self.cfg, self.bank, self.probes, only="sonnet-bugfixer", limit=1)[0]
        text = run_night.conversation_prompt(c)
        self.assertIn(c["seed_q"], text)
        self.assertIn("codemap_feedback", text)
        cfg = run_night.mcp_config(c, "https://x.example/", "tok")
        self.assertEqual(cfg["mcpServers"]["codemap"]["url"], "https://x.example/mcp")
        self.assertEqual(cfg["mcpServers"]["codemap"]["headers"]["X-CodeMap-User"], "sonnet-bugfixer")
        cb = dict(c, mode="baseline")
        self.assertEqual(run_night.mcp_config(cb, "u", "t"), {"mcpServers": {}})
        self.assertIn("BASELINE", run_night.conversation_prompt(cb))
        with tempfile.TemporaryDirectory() as d:
            sf = run_night.system_file(c, d)
            body = open(sf, encoding="utf-8").read()
            self.assertIn("sonnet-bugfixer", body)
            self.assertIn("Rating anchors", body)

    def test_otel_env_only_with_a_token(self):
        self.assertEqual(run_night.otel_env("persona", "haiku-pm", "2026-09-17", env={}), {})
        e = run_night.otel_env("persona", "haiku-pm", "2026-09-17", env={"GRAFANA_CLOUD_ALLOY_TOKEN": "t"})
        self.assertIn("persona=haiku-pm", e["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertEqual(e["OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE"], "cumulative")

    def test_report_parser(self):
        txt = 'done\n```json\n{"conversations": [{"context_id": "c", "turns": [{"request_id": "r", "rating": 4, "verified": true}]}], "misses": []}\n```'
        self.assertEqual(run_night.parse_report(txt)["conversations"][0]["turns"][0]["rating"], 4)
        self.assertIsNone(run_night.parse_report("no block"))

    def test_pairs_first_per_persona(self):
        cfg = dict(self.cfg, baseline_share=0.5)
        convs = run_night.plan("2026-09-20", cfg, self.bank, self.probes, seed=20)
        for pid in {c["persona"] for c in convs}:
            mine = [c for c in convs if c["persona"] == pid]
            paired = {c["seed_id"] for c in mine if c["mode"] == "baseline"}
            flags = [c["seed_id"] in paired for c in mine]
            self.assertEqual(flags, sorted(flags, reverse=True), (pid, flags))  # paired block, then the rest
            for i, c in enumerate(mine):
                if c["mode"] == "baseline":
                    self.assertEqual((mine[i + 1]["mode"], mine[i + 1]["seed_id"]), ("codemap", c["seed_id"]))

    def test_conversations_override_replaces_the_per_persona_count(self):
        cfg = dict(self.cfg, baseline_share=1.0)
        convs = run_night.plan("2026-09-20", cfg, self.bank, self.probes, seed=1, conversations=8)
        for p in cfg["personas"]:
            mine = [c for c in convs if c["persona"] == p["id"] and not c.get("paired_with")]
            self.assertEqual(len(mine), min(8, len(run_night.slice_for(p, self.bank, self.probes))), p["id"])
        self.assertEqual(sum(1 for c in convs if c["mode"] == "baseline"),
                         sum(1 for c in convs if c.get("paired_with") == "baseline"))

    def test_partner_reserve(self):
        self.assertEqual(run_night.partner_reserve({"spent": 0, "requests": 0}, 4), 60.0)
        self.assertEqual(run_night.partner_reserve({"spent": 498.4579, "requests": 9}, 3), round(498.4579 / 9 * 3, 2))
        self.assertEqual(run_night.partner_reserve({"spent": 10, "requests": 5}, 3), 24.0)  # floor of 8 per ask

    def test_baseline_share_override_reaches_the_plan(self):
        cfg = dict(self.cfg, baseline_share=0.5)
        half = run_night.plan("2026-09-18", cfg, self.bank, self.probes, seed=2)
        base = sum(1 for c in half if c["mode"] == "baseline")
        self.assertGreaterEqual(base, 5)  # 18 draws at share 0.5 (first of each persona is never a baseline)

    def test_every_baseline_has_a_codemap_partner_on_the_same_seed(self):
        convs = run_night.plan("2026-09-17", self.cfg, self.bank, self.probes, seed=3)
        base = [c for c in convs if c["mode"] == "baseline"]
        self.assertTrue(base, "the plan should hold at least one baseline")
        cm = {(c["persona"], c["seed_id"]) for c in convs if c["mode"] == "codemap"}
        for c in base:
            self.assertIn((c["persona"], c["seed_id"]), cm)


class _Proc:
    def __init__(self, stdout):
        self.stdout, self.stderr, self.returncode = stdout, "", 0


class NightRunTests(unittest.TestCase):
    def _args(self, **kw):
        base = dict(date="2026-09-17", seed=3, persona=None, limit=1, haiku_only=True, max_credits=3000,
                    max_turns=4, timeout=10, dry_run=False, sync_agents=False)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_rate_limit_ends_the_night_partial(self):
        calls = []

        def runner(cmd, env, cwd, timeout):
            calls.append(cmd)
            if len(calls) == 1:
                return _Proc(json.dumps({"type": "result", "is_error": False, "result": "ok\n```json\n"
                                         '{"conversations": [{"context_id": "c", "turns": [{"request_id": "r", "rating": 5, "verified": true}]}], "misses": ["backend/x.yml"]}\n```',
                                         "session_id": "s1", "num_turns": 3, "usage": {"input_tokens": 5, "output_tokens": 50}}))
            return _Proc(json.dumps({"type": "result", "is_error": True, "result": "Rate limit reached"}))

        def http(method, url, body):
            if "/budget" in url:
                return {"spent": 0.0, "remaining": 200.0, "exhausted": False}
            return []
        # runs dir is redirected by monkeypatching HERE's runs path through a temp copy
        with tempfile.TemporaryDirectory() as d:
            real = run_night.HERE
            run_night.HERE = d
            os.makedirs(os.path.join(d, "roles"))
            for name in os.listdir(os.path.join(real, "roles")):
                open(os.path.join(d, "roles", name), "w", encoding="utf-8").write(open(os.path.join(real, "roles", name), encoding="utf-8").read())
            open(os.path.join(d, "personas.json"), "w", encoding="utf-8").write(open(os.path.join(real, "personas.json"), encoding="utf-8").read())
            try:
                s = run_night.run(self._args(), runner=runner, http=http, env={"CODEMAP_URL": "http://x", "CODEMAP_TOKEN": "t"})
            finally:
                run_night.HERE = real
            self.assertTrue(s["partial"])
            self.assertEqual(s["stopped_reason"], "rate limited")
            self.assertEqual(s["conversations"], 2)
            first = next(iter(s["by_persona"].values()))
            self.assertEqual(first["ratings"], [5])
            self.assertEqual(first["misses"], 1)
            rows = [json.loads(l) for l in open(os.path.join(d, "runs", "2026-09-17.jsonl"), encoding="utf-8")]
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[1]["rate_limited"])
            # the persona call carried the MCP and the role file; the key never leaked
            cmd = calls[0]
            self.assertIn("mcp__codemap__*,Read,Grep,Glob", cmd)
            self.assertIn("--append-system-prompt-file", cmd) if "--append-system-prompt-file" in cmd else self.assertIn("--system-prompt-file", cmd)

    def test_baseline_skipped_when_the_partner_is_unaffordable(self):
        calls = []
        runner = lambda cmd, env, cwd, timeout: calls.append(cmd) or _Proc(json.dumps({"type": "result", "is_error": False, "result": "x"}))
        # 20 credits left at 25 per ask: no baseline can be paired today, and the partner is not run either
        http = lambda method, url, body: {"spent": 180.0, "requests": 7, "remaining": 20.0, "exhausted": False} if "/budget" in url else []
        with tempfile.TemporaryDirectory() as d:
            real = run_night.HERE
            run_night.HERE = d
            os.makedirs(os.path.join(d, "roles"))
            for name in os.listdir(os.path.join(real, "roles")):
                open(os.path.join(d, "roles", name), "w", encoding="utf-8").write(open(os.path.join(real, "roles", name), encoding="utf-8").read())
            open(os.path.join(d, "personas.json"), "w", encoding="utf-8").write(open(os.path.join(real, "personas.json"), encoding="utf-8").read())
            try:
                run_night.run(self._args(persona="haiku-pm", limit=3, baseline_share=1.0), runner=runner, http=http,
                              env={"CODEMAP_URL": "http://x", "CODEMAP_TOKEN": "t"})
            finally:
                run_night.HERE = real
            rows = [json.loads(l) for l in open(os.path.join(d, "runs", "2026-09-17.jsonl"), encoding="utf-8")]
        skipped = [r.get("skipped") for r in rows]
        self.assertIn("no_budget_for_partner", skipped)
        self.assertIn("baseline_skipped", skipped)
        self.assertFalse(any(r.get("mode") == "baseline" and not r.get("skipped") for r in rows))
        self.assertEqual(len(calls), 1)  # only the unpaired CodeMap conversation ran

    def test_exhausted_budget_skips_without_a_call(self):
        calls = []
        runner = lambda cmd, env, cwd, timeout: calls.append(cmd) or _Proc(json.dumps({"type": "result", "is_error": False, "result": "x"}))
        http = lambda method, url, body: {"spent": 200.0, "remaining": 0.0, "exhausted": True} if "/budget" in url else []
        with tempfile.TemporaryDirectory() as d:
            real = run_night.HERE
            run_night.HERE = d
            os.makedirs(os.path.join(d, "roles"))
            for name in os.listdir(os.path.join(real, "roles")):
                open(os.path.join(d, "roles", name), "w", encoding="utf-8").write(open(os.path.join(real, "roles", name), encoding="utf-8").read())
            open(os.path.join(d, "personas.json"), "w", encoding="utf-8").write(open(os.path.join(real, "personas.json"), encoding="utf-8").read())
            try:
                s = run_night.run(self._args(persona="haiku-pm"), runner=runner, http=http, env={"CODEMAP_URL": "http://x", "CODEMAP_TOKEN": "t"})
            finally:
                run_night.HERE = real
        self.assertEqual(calls, [])
        self.assertEqual(s["by_persona"]["haiku-pm"]["errors"], 1)

    def test_dry_run_and_agent_sync(self):
        s = run_night.run(self._args(dry_run=True, haiku_only=False, limit=None))
        self.assertEqual(s["planned"], 21)  # 18 conversations + a CodeMap partner for each of the 3 baselines
        n = run_night.sync_agents()
        self.assertEqual(n, 6)
        body = open(os.path.join(R, ".claude", "agents", "opus-reviewer.md"), encoding="utf-8").read()
        self.assertTrue(body.startswith("---\nname: opus-reviewer"))
        self.assertIn("model: opus", body)


if __name__ == "__main__":
    unittest.main()
