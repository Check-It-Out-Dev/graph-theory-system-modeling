"""Dashboards as code: six Grafana dashboards generated deterministically into observability/grafana/.

    python tools/grafana/build_dashboards.py [--check]

Public-dashboard laws (learned on the earlier arcs): no template variables (a public panel with $var
returns zero rows), explicit panel ids, `max_over_time(x[$__range])` on counters the server pushes as
running totals, `${DS_PROMETHEUS}` as the only input. Currency appears nowhere: credits and tokens only.
"""

import argparse
import json
import os
import sys

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(R, "observability", "grafana")
sys.path.insert(0, R)

DS = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}
INPUTS = [{"name": "DS_PROMETHEUS", "label": "Prometheus", "description": "", "type": "datasource", "pluginId": "prometheus",
           "pluginName": "Prometheus"}]


class Dash:
    def __init__(self, uid, title, description, tags):
        self.uid, self.title, self.description, self.tags = uid, title, description, tags
        self.panels = []
        self._id = 0
        self._y = 0
        self._x = 0
        self._row_h = 0

    def _place(self, w, h):
        """Left to right; wrap to a new line when the 24-column row is full; y advances by the tallest panel."""
        if self._x + w > 24:
            self._x, self._y, self._row_h = 0, self._y + self._row_h, 0
        pos = {"x": self._x, "y": self._y, "w": w, "h": h}
        self._x += w
        self._row_h = max(self._row_h, h)
        return pos

    def row(self, title):
        if self._x:
            self._y += self._row_h
        self._x, self._row_h = 0, 0
        self._id += 1
        self.panels.append({"id": self._id, "type": "row", "title": title, "collapsed": False,
                            "gridPos": {"x": 0, "y": self._y, "w": 24, "h": 1}, "panels": []})
        self._y += 1

    def panel(self, ptype, title, targets, w=8, h=8, unit=None, options=None, overrides=None, description=None, text=None):
        self._id += 1
        p = {"id": self._id, "type": ptype, "title": title, "gridPos": self._place(w, h), "datasource": DS,
             "targets": [dict({"refId": chr(65 + i), "datasource": DS}, **t) for i, t in enumerate(targets)],
             "fieldConfig": {"defaults": {}, "overrides": overrides or []}, "options": options or {}}
        if unit:
            p["fieldConfig"]["defaults"]["unit"] = unit
        if description:
            p["description"] = description
        if ptype == "text":
            p["options"] = {"mode": "markdown", "content": text or ""}
            p.pop("targets")
            p.pop("datasource")
        if ptype == "stat":
            p["options"].setdefault("reduceOptions", {"calcs": ["lastNotNull"], "fields": "", "values": False})
            p["options"].setdefault("colorMode", "value")
            p["options"].setdefault("graphMode", "none")
        if ptype == "timeseries":
            p["options"].setdefault("legend", {"displayMode": "list", "placement": "bottom", "showLegend": True})
            # one sample per night: without points a single night draws nothing
            p["fieldConfig"]["defaults"].setdefault("custom", {"lineWidth": 2, "fillOpacity": 8, "spanNulls": True,
                                                               "showPoints": "always", "pointSize": 7})
        if ptype == "bargauge":
            p["options"].setdefault("reduceOptions", {"calcs": ["lastNotNull"], "fields": "", "values": False})
            p["options"].setdefault("orientation", "horizontal")
            p["options"].setdefault("displayMode", "gradient")
        self.panels.append(p)
        return p

    def build(self):
        return {"__inputs": INPUTS, "__requires": [], "uid": self.uid, "title": self.title, "description": self.description,
                "tags": self.tags, "timezone": "utc", "schemaVersion": 39, "version": 1, "editable": False, "graphTooltip": 1,
                "time": {"from": "now-14d", "to": "now"}, "refresh": "5m", "templating": {"list": []},
                "annotations": {"list": [{"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"}, "enable": True,
                                          "hide": True, "iconColor": "rgba(0, 211, 255, 1)", "name": "Annotations & Alerts", "type": "dashboard"},
                                         {"datasource": {"type": "grafana", "uid": "-- Grafana --"}, "enable": True, "hide": False,
                                          "iconColor": "#F2CC0C", "name": "prompt promotions", "target": {"limit": 100, "matchAny": True,
                                                                                                          "tags": ["codemap", "promotion"], "type": "tags"}}]},
                "links": [{"title": "CodeMap Remote — the design", "type": "link", "url": "https://github.com/Check-It-Out-Dev/graph-theory-system-modeling/blob/main/applications/CodeMap/docs/07-ai-quality-governance.md", "targetBlank": True}],
                "panels": self.panels}


def tot(metric, by=None, extra=""):
    """A running-total counter pushed by the server, as its value over the range."""
    inner = f"max_over_time({metric}{extra}[$__range])"
    return f"sum by ({by}) ({inner})" if by else f"sum({inner})"


def q(metric, key=None):
    """A nightly quality gauge (one point per night)."""
    return f"{metric}{{key=\"{key}\"}}" if key else metric


def rate_card_markdown():
    from remote import credits
    card = credits.RateCard()
    return ("### Credit rate card (no currency)\n\n" + card.doc["note"] + "\n\n" + card.table() +
            "\n\nA navigator answer costs 20–30 credits; a FAQ hit and an engine step cost nothing; a semantic search costs 2.")


def build_all():
    dashes = []

    d = Dash("codemap-ai-system", "CodeMap · AI system", "The served MCP as a system: traffic, outcomes, quality, versions. Public.", ["codemap", "public"])
    d.row("Tonight and this range")
    d.panel("stat", "Navigator answers", [{"expr": tot("codemap_requests_total", extra='{event_type="ask",tier=~"nav-.*",terminal="answer"}'), "legendFormat": "answers"}], w=4, h=5)
    d.panel("stat", "Abstentions", [{"expr": tot("codemap_requests_total", extra='{event_type="ask",tier=~"nav-.*",terminal="abstain"}'), "legendFormat": "abstain"}], w=4, h=5)
    d.panel("stat", "FAQ hits (free)", [{"expr": tot("codemap_requests_total", extra='{event_type="ask",tier="faq"}'), "legendFormat": "faq"}], w=4, h=5)
    d.panel("stat", "Engine steps (free)", [{"expr": tot("codemap_requests_total", extra='{event_type="step"}'), "legendFormat": "steps"}], w=4, h=5)
    d.panel("stat", "Misses reported", [{"expr": tot("codemap_miss_total"), "legendFormat": "misses"}], w=4, h=5)
    d.panel("stat", "Budget refusals", [{"expr": tot("codemap_budget_refusals_total"), "legendFormat": "refusals"}], w=4, h=5)
    d.row("Quality per night (judge, calibrated)")
    d.panel("timeseries", "Grounded · correct · helpful (share of judged answers)",
            [{"expr": q("codemap_quality_grounded_rate"), "legendFormat": "grounded"},
             {"expr": q("codemap_quality_correct_rate"), "legendFormat": "correct"},
             {"expr": q("codemap_quality_helpful_rate"), "legendFormat": "helpful"},
             {"expr": q("codemap_quality_oracle_success_rate"), "legendFormat": "oracle success"}], w=12, h=8, unit="percentunit")
    d.panel("timeseries", "Graph coverage of the checkout (saturation): indexed ∩ eligible / eligible, per repository",
            [{"expr": q("codemap_graph_coverage_ratio", "backend"), "legendFormat": "backend"},
             {"expr": q("codemap_graph_coverage_ratio", "frontend"), "legendFormat": "frontend"}], w=12, h=8, unit="percentunit")
    d.panel("timeseries", "Version drift after a pack decision (bank rows answering differently, not invalidated on purpose)",
            [{"expr": q("codemap_quality_version_drift_rate"), "legendFormat": "drift rate"}], w=12, h=8, unit="percentunit")
    d.panel("timeseries", "Abstentions: all · honest · false",
            [{"expr": q("codemap_quality_abstention_rate", "all"), "legendFormat": "all"},
             {"expr": q("codemap_quality_abstention_rate", "honest"), "legendFormat": "honest"},
             {"expr": q("codemap_quality_abstention_rate", "false"), "legendFormat": "false"}], w=12, h=8, unit="percentunit")
    d.row("Behaviour")
    d.panel("timeseries", "Engine steps per answer (mean)", [{"expr": q("codemap_quality_steps_mean"), "legendFormat": "steps"}], w=8, h=7)
    d.panel("timeseries", "Latency p95 by tier (ms)", [{"expr": q("codemap_quality_latency_ms_p95"), "legendFormat": "{{key}}"}], w=8, h=7, unit="ms")
    d.panel("table", "Versions in service", [{"expr": "codemap_info", "format": "table", "instant": True}], w=8, h=7,
            options={"showHeader": True}, description="prompt_version = sha16 of the built navigator prompt; pack_version = the Release")
    dashes.append(d)

    d = Dash("codemap-happiness", "CodeMap · Happiness", "What the users say after verifying a pointer. Public.", ["codemap", "public"])
    d.row("Ratings")
    d.panel("stat", "Rating mean (last night)", [{"expr": q("codemap_quality_rating_mean"), "legendFormat": "mean"}], w=6, h=5)
    d.panel("stat", "Rated ≥ 4", [{"expr": q("codemap_quality_rating_ge4_rate"), "legendFormat": "≥4"}], w=6, h=5, unit="percentunit")
    d.panel("stat", "Pointer verified before rating", [{"expr": q("codemap_quality_pointer_verified_rate"), "legendFormat": "verified"}], w=6, h=5, unit="percentunit")
    d.panel("stat", "Ratings received", [{"expr": tot("codemap_feedback_total"), "legendFormat": "n"}], w=6, h=5)
    d.row("Distribution and reasons")
    d.panel("bargauge", "Ratings by value", [{"expr": tot("codemap_feedback_total", by="rating"), "legendFormat": "★ {{rating}}"}], w=8, h=8)
    d.panel("bargauge", "Tags", [{"expr": tot("codemap_feedback_tags_total", by="tag"), "legendFormat": "{{tag}}"}], w=8, h=8)
    d.panel("bargauge", "Ratings by tier", [{"expr": tot("codemap_feedback_total", by="tier"), "legendFormat": "{{tier}}"}], w=8, h=8)
    d.row("Over nights")
    d.panel("timeseries", "Rating mean and share ≥ 4", [{"expr": q("codemap_quality_rating_mean"), "legendFormat": "mean (1-5)"},
                                                        {"expr": q("codemap_quality_rating_ge4_rate"), "legendFormat": "share ≥4"}], w=24, h=8)
    dashes.append(d)

    d = Dash("codemap-usage-tokens", "CodeMap · Usage & tokens", "Tokens on the server side (the navigator) and on the users' side (Claude Code personas, judge, reflector). Public.", ["codemap", "public"])
    d.row("Server: the navigator's tokens")
    d.panel("bargauge", "Tokens by type (navigator)", [{"expr": tot("codemap_tokens_total", by="type", extra='{tier=~"nav-.*"}'), "legendFormat": "{{type}}"}], w=12, h=8)
    d.panel("stat", "Cache read ratio (last night)", [{"expr": q("codemap_quality_cache_read_ratio"), "legendFormat": "cache"}], w=6, h=8, unit="percentunit",
            description="cached / (cached + created + fresh). Follow-ups resume the Claude session, so the second turn is mostly cache.")
    d.panel("stat", "Requests by user", [{"expr": tot("codemap_requests_total", by="user", extra='{event_type="ask"}'), "legendFormat": "{{user}}"}], w=6, h=8,
            options={"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": True}, "textMode": "value_and_name"})
    d.row("Users: Claude Code's own telemetry, by role and persona (subscription tokens)")
    d.panel("timeseries", "Tokens per role", [{"expr": "sum by (role) (claude_code_token_usage_tokens_total)", "legendFormat": "{{role}}"}], w=12, h=8)
    d.panel("bargauge", "Tokens per persona", [{"expr": "sum by (persona) (claude_code_token_usage_tokens_total{role=\"persona\"})", "legendFormat": "{{persona}}"}], w=12, h=8)
    d.panel("timeseries", "Tokens by type (all roles)", [{"expr": "sum by (type) (claude_code_token_usage_tokens_total)", "legendFormat": "{{type}}"}], w=24, h=7)
    dashes.append(d)

    d = Dash("codemap-credit-rate-card", "CodeMap · Credit rate card", "What an answer costs, in credits (relative units, no currency). Public.", ["codemap", "public"])
    d.row("The card")
    d.panel("text", "Rate card", [], w=12, h=12, text=rate_card_markdown())
    d.panel("stat", "Credits per answer (last night)", [{"expr": q("codemap_quality_credits_per_answer"), "legendFormat": "per answer"}], w=6, h=6)
    d.panel("stat", "Credits per CORRECT answer", [{"expr": q("codemap_quality_credits_per_correct_answer"), "legendFormat": "per correct"}], w=6, h=6)
    d.panel("bargauge", "Credits by tier", [{"expr": tot("codemap_credits_total", by="tier"), "legendFormat": "{{tier}}"}], w=12, h=6)
    dashes.append(d)

    d = Dash("codemap-quality-over-time", "CodeMap · Quality over time", "Judge calibration and the rates the README claims, night by night; prompt promotions as annotations. Public.", ["codemap", "public"])
    d.row("Is the judge worth believing?")
    d.panel("timeseries", "Cohen's κ: judge vs oracle · judge vs users", [{"expr": q("codemap_quality_judge_kappa", "oracle"), "legendFormat": "κ oracle (gate 0.6)"},
                                                                          {"expr": q("codemap_quality_judge_kappa", "human"), "legendFormat": "κ users"},
                                                                          {"expr": q("codemap_quality_judge_kappa", "anchor_now"), "legendFormat": "κ anchors"}], w=12, h=8,
            overrides=[{"matcher": {"id": "byName", "options": "κ oracle (gate 0.6)"}, "properties": [{"id": "thresholds", "value": {"mode": "absolute", "steps": [{"color": "red", "value": None}, {"color": "green", "value": 0.6}]}}]}])
    d.panel("timeseries", "Cross-family agreement (Claude judge vs Qwen reranker) · disputes", [{"expr": q("codemap_quality_judge_agreement"), "legendFormat": "agreement"},
                                                                                                 {"expr": q("codemap_quality_disputes_total"), "legendFormat": "disputes"}], w=12, h=8)
    d.row("The rates")
    d.panel("timeseries", "Grounded · correct · helpful · rating ≥4", [{"expr": q("codemap_quality_grounded_rate"), "legendFormat": "grounded"},
                                                                      {"expr": q("codemap_quality_correct_rate"), "legendFormat": "correct"},
                                                                      {"expr": q("codemap_quality_helpful_rate"), "legendFormat": "helpful"},
                                                                      {"expr": q("codemap_quality_rating_ge4_rate"), "legendFormat": "rating ≥4"}], w=24, h=9, unit="percentunit")
    dashes.append(d)

    d = Dash("codemap-budget-burn-gains", "CodeMap · Budget burn & gains", "Each persona's daily credits against its cap, and what CodeMap saved against the no-CodeMap baseline. Public.", ["codemap", "public"])
    d.row("Budgets today")
    d.panel("bargauge", "Credits remaining today, by user", [{"expr": "codemap_budget_remaining", "legendFormat": "{{user}}"}], w=12, h=9,
            options={"orientation": "horizontal", "displayMode": "lcd", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}})
    d.panel("bargauge", "Credits spent (range), by user", [{"expr": tot("codemap_credits_total", by="user"), "legendFormat": "{{user}}"}], w=12, h=9)
    d.row("Gains vs the no-CodeMap baseline (same persona, same question, seeded)")
    d.panel("stat", "Token ratio baseline / CodeMap", [{"expr": q("codemap_quality_gain", "tokens_ratio_mean"), "legendFormat": "×"}], w=6, h=6, description="> 1 means the baseline burned more tokens to answer the same question")
    d.panel("stat", "Turns saved (mean)", [{"expr": q("codemap_quality_gain", "turns_delta_mean"), "legendFormat": "turns"}], w=6, h=6)
    d.panel("stat", "Seconds saved (mean)", [{"expr": q("codemap_quality_gain", "seconds_delta_mean"), "legendFormat": "s"}], w=6, h=6, unit="s")
    d.panel("stat", "Paired conversations", [{"expr": q("codemap_quality_gain", "n_pairs"), "legendFormat": "pairs"}], w=6, h=6)
    d.panel("timeseries", "Self-reported minutes saved (mean per night)", [{"expr": q("codemap_quality_gain", "minutes_saved_estimate_mean"), "legendFormat": "minutes"}], w=24, h=7, unit="m")
    dashes.append(d)
    return dashes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    stale = 0
    for d in build_all():
        doc = d.build()
        text = json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        path = os.path.join(OUT, f"{d.uid}.json")
        if a.check:
            cur = open(path, encoding="utf-8").read().replace("\r\n", "\n") if os.path.exists(path) else ""
            if cur != text:
                print(f"STALE {path}")
                stale += 1
            continue
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"wrote {path} ({len(doc['panels'])} panels)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
