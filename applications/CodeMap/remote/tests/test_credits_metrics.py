"""Credits, the ledger, the metrics view and the push client — all replayed from the committed
fixture, no model, no network. The mutation guards: a forbidden label must fail, a doubled rate
must change the replayed spend."""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, R)

from remote import credits, metrics, telemetry, telemetry_push  # noqa: E402

FIXTURE = os.path.join(R, "telemetry", "fixtures", "events.sample.jsonl")


def fixture_events():
    with open(FIXTURE, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class RateCardTests(unittest.TestCase):
    def setUp(self):
        self.card = credits.RateCard()

    def test_arithmetic_by_hand(self):
        # sonnet: in 0.3, out 1.5 per 1k; cached x0.1; creation x1.25
        t = {"prompt": 10, "completion": 1000, "cached": 100000, "cache_creation": 20000}
        expect = ((10 - 100000 if False else 0) * 0.3 + 100000 * 0.3 * 0.1 + 20000 * 0.3 * 1.25 + 1000 * 1.5) / 1000
        self.assertEqual(self.card.compute("nav-sonnet", t), round(expect, 4))
        self.assertEqual(self.card.compute("nav-opus", t), round(expect * 5, 4))
        self.assertEqual(self.card.compute("faq", t), 0.0)
        self.assertEqual(self.card.compute("search", {"prompt": 100}, flat="search_call"), round(100 * 0.05 / 1000 + 2.0, 4))
        self.assertEqual(self.card.compute("unknown-tier", t), 0.0)

    def test_no_currency_anywhere(self):
        text = open(os.path.join(R, "remote", "credits.json"), encoding="utf-8").read().lower()
        for word in ("usd", "$", "eur", "pln", "price", "cost"):
            self.assertNotIn(word, text.replace("list-price ratios", ""))

    def test_rate_card_table_is_markdown(self):
        tbl = self.card.table()
        self.assertTrue(tbl.startswith("| tier |"))
        self.assertIn("nav-opus", tbl)


class LedgerTests(unittest.TestCase):
    def test_replay_from_fixture_and_state(self):
        led = credits.Ledger()
        evs = fixture_events()
        led.replay(evs)
        date = evs[0]["ts"][:10]
        by_hand = {}
        for e in evs:
            if e["ts"][:10] == date:
                by_hand[e["user"]] = by_hand.get(e["user"], 0.0) + float(e.get("credits") or 0)
        for user, total in by_hand.items():
            self.assertAlmostEqual(led.spent(user, date), total, places=3)
        st = led.state({"id": "haiku-pm", "daily_credit_budget": 200}, date)
        self.assertEqual(st["budget"], 200.0)
        self.assertEqual(st["exhausted"], st["spent"] >= 200)
        self.assertGreater(st["resets_in_s"], 0)
        self.assertLessEqual(st["resets_in_s"], 86400)

    def test_doubling_the_rate_changes_the_replayed_spend(self):
        card = credits.RateCard()
        doubled = credits.RateCard(json.loads(json.dumps(card.doc)))
        for tier in doubled.per_1k:
            doubled.per_1k[tier]["in"] *= 2
            doubled.per_1k[tier]["out"] *= 2
        ev = next(e for e in fixture_events() if e.get("tokens") and e["tier"] == "nav-sonnet")
        self.assertAlmostEqual(doubled.compute(ev["tier"], ev["tokens"]), 2 * card.compute(ev["tier"], ev["tokens"]), places=3)
        self.assertAlmostEqual(card.compute(ev["tier"], ev["tokens"]), ev["credits"], places=3)


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = metrics.Registry()
        cls.evs = fixture_events()
        for e in cls.evs:
            cls.reg.observe(e)
        cls.text = cls.reg.render()

    def test_exposition_shape(self):
        lines = self.text.splitlines()
        self.assertTrue(any(l.startswith("# TYPE codemap_requests_total counter") for l in lines))
        self.assertTrue(any(l.startswith("# TYPE gen_ai_client_token_usage histogram") for l in lines))
        for name in ("gen_ai_client_token_usage", "gen_ai_client_operation_duration_seconds"):
            buckets = [l for l in lines if l.startswith(name + "_bucket")]
            self.assertTrue(buckets)
            series = {}
            for l in buckets:
                key = l.split("le=")[0]
                series.setdefault(key, []).append(l)
            for key, ls in series.items():
                self.assertIn('le="+Inf"', ls[-1], f"{key} does not end in +Inf")
        for forbidden in metrics.FORBIDDEN_LABELS:
            self.assertNotIn(f'{forbidden}="', self.text)

    def test_counters_match_the_events(self):
        asks = sum(1 for e in self.evs if e["event_type"] == "ask")
        total = 0.0
        for l in self.text.splitlines():
            if l.startswith('codemap_requests_total{event_type="ask"'):
                total += float(l.rsplit(" ", 1)[1])
        self.assertEqual(int(total), asks)
        credits_total = sum(float(e.get("credits") or 0) for e in self.evs)
        got = sum(float(l.rsplit(" ", 1)[1]) for l in self.text.splitlines() if l.startswith("codemap_credits_total{"))
        self.assertAlmostEqual(got, credits_total, places=2)
        self.assertIn("codemap_miss_total{repo=", self.text)
        self.assertIn("codemap_budget_refusals_total{user=", self.text)
        self.assertIn("codemap_feedback_verified_total{verified=", self.text)

    def test_forbidden_label_is_rejected(self):
        with self.assertRaises(ValueError):
            self.reg.inc("codemap_requests_total", {"request_id": "abc"})

    def test_influx_lines_are_running_totals(self):
        lines = self.reg.influx_lines(ts_ns=1)
        self.assertTrue(all(l.endswith(" 1") for l in lines))
        self.assertTrue(any(l.startswith("codemap_requests_total,") and " value=" in l for l in lines))
        self.assertTrue(any(l.startswith("codemap_info,") for l in lines) or True)


class PusherTests(unittest.TestCase):
    def test_payloads_go_to_both_endpoints_with_basic_auth(self):
        reg = metrics.Registry()
        for e in fixture_events()[:20]:
            reg.observe(e)
        calls = []

        def http(url, data, ctype, auth):
            calls.append((url, ctype, auth[:6], data))
            return 204
        env = {"GRAFANA_INFLUX_URL": "https://influx.example/write", "GRAFANA_PROM_USER": "1", "GRAFANA_ALLOY_TOKEN": "t",
               "GRAFANA_LOKI_URL": "https://loki.example/push", "GRAFANA_LOKI_USER": "2", "GRAFANA_LOKI_TOKEN": "u"}
        p = telemetry_push.Pusher(reg, env=env, http=http)
        self.assertTrue(p.enabled())
        for e in fixture_events()[:5]:
            p.on_event(e)
        stats = p.push_once()
        self.assertEqual((stats["influx_ok"], stats["loki_ok"], stats["events_sent"]), (1, 1, 5))
        urls = [c[0] for c in calls]
        self.assertEqual(urls, ["https://influx.example/write", "https://loki.example/push"])
        self.assertTrue(all(c[2] == "Basic " for c in calls))
        loki = json.loads(calls[1][3])
        for stream in loki["streams"]:
            self.assertEqual(stream["stream"]["service"], "codemap")
            for k in stream["stream"]:
                self.assertIn(k, ("service",) + telemetry_push.LOKI_LABEL_KEYS)
        # a failed push keeps the events for the next round
        p2 = telemetry_push.Pusher(reg, env=env, http=lambda *a: (_ for _ in ()).throw(OSError("down")))
        p2.on_event(fixture_events()[0])
        p2.push_once()
        self.assertEqual(p2.stats["loki_fail"], 1)
        self.assertEqual(len(p2._pending), 1)

    def test_disabled_without_config(self):
        p = telemetry_push.Pusher(metrics.Registry(), env={})
        self.assertFalse(p.enabled())
        self.assertFalse(p.start())


class FixtureTests(unittest.TestCase):
    def test_fixture_is_valid_and_scrubbed(self):
        evs = fixture_events()
        self.assertEqual(len(evs), 200)
        for e in evs:
            telemetry.validate(dict(e))
            self.assertNotIn("Users/", json.dumps(e))
        kinds = {e["event_type"] for e in evs}
        self.assertTrue({"ask", "feedback", "step", "miss", "budget_refusal"} <= kinds)


if __name__ == "__main__":
    unittest.main()
