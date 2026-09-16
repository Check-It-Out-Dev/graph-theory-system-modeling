"""A Prometheus view of the events file, rebuilt at boot and updated on every emit.

Names follow the OpenTelemetry GenAI semantic conventions where one exists
(`gen_ai_client_token_usage`, `gen_ai_client_operation_duration_seconds`) and `codemap_*` for the
rest. Cardinality law: `user` only on counters and gauges, never on histograms; never
request_id / context_id / q as a label. Text exposition 0.0.4 with HELP/TYPE lines and cumulative
`le` buckets ending in +Inf.
"""

import math
import threading

TOKEN_BUCKETS = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576)
SECONDS_BUCKETS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92, 163.84)
FORBIDDEN_LABELS = ("request_id", "context_id", "q", "session_id", "answer", "comment")
INF = "+Inf"


def _esc(v):
    return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(d):
    if not d:
        return ""
    return "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in sorted(d.items())) + "}"


class Registry:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters = {}   # (name, labels-tuple) -> float
        self.gauges = {}
        self.hists = {}      # (name, labels-tuple) -> {"buckets": [..counts..], "sum": x, "count": n, "edges": (...)}
        self.help = {}
        self.types = {}
        self._doc("codemap_requests_total", "counter", "Tool calls by type, tier, terminal and user.")
        self._doc("codemap_credits_total", "counter", "Credits charged (no currency), by user and tier.")
        self._doc("codemap_tokens_total", "counter", "Model tokens by tier and type (prompt, completion, cached, cache_creation).")
        self._doc("codemap_steps_total", "counter", "Engine steps taken by navigators, by tier.")
        self._doc("codemap_feedback_total", "counter", "Ratings received, by tier and rating.")
        self._doc("codemap_feedback_tags_total", "counter", "Feedback tags, by tag.")
        self._doc("codemap_feedback_verified_total", "counter", "Feedback rows by whether a pointer was verified.")
        self._doc("codemap_miss_total", "counter", "Files found without the graph (saturation backlog), by repo.")
        self._doc("codemap_budget_refusals_total", "counter", "Requests refused for an exhausted daily budget, by user.")
        self._doc("codemap_budget_remaining", "gauge", "Credits left today, by user.")
        self._doc("codemap_info", "gauge", "Versions in service (always 1).")
        self._doc("gen_ai_client_token_usage", "histogram", "Tokens per request (OTel GenAI semconv), by model, token type and tier.")
        self._doc("gen_ai_client_operation_duration_seconds", "histogram", "Request duration (OTel GenAI semconv), by tier.")

    def _doc(self, name, typ, help_):
        self.help[name] = help_
        self.types[name] = typ

    # ------------------------------------------------------------------ writes
    def inc(self, name, labels, value=1.0):
        self._check(labels)
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self.counters[key] = self.counters.get(key, 0.0) + float(value)

    def set(self, name, labels, value):
        self._check(labels)
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self.gauges[key] = float(value)

    def observe_hist(self, name, labels, value, edges):
        self._check(labels)
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            h = self.hists.get(key)
            if h is None:
                h = self.hists[key] = {"buckets": [0] * (len(edges) + 1), "sum": 0.0, "count": 0, "edges": edges}
            v = float(value)
            for i, e in enumerate(edges):
                if v <= e:
                    h["buckets"][i] += 1
            h["buckets"][-1] += 1
            h["sum"] += v
            h["count"] += 1

    @staticmethod
    def _check(labels):
        for k in labels:
            if k in FORBIDDEN_LABELS:
                raise ValueError(f"label {k} is forbidden on metrics (cardinality law)")

    # ------------------------------------------------------------------ events -> metrics
    def observe(self, ev):
        et = ev.get("event_type")
        tier = ev.get("tier") or "none"
        user = ev.get("user") or "?"
        term = ev.get("terminal") or "none"
        self.inc("codemap_requests_total", {"event_type": et, "tier": tier, "terminal": term, "user": user})
        c = float(ev.get("credits") or 0)
        if c:
            self.inc("codemap_credits_total", {"user": user, "tier": tier}, c)
        tokens = ev.get("tokens") or {}
        for typ in ("prompt", "completion", "cached", "cache_creation"):
            v = tokens.get(typ)
            if v:
                self.inc("codemap_tokens_total", {"tier": tier, "type": typ}, v)
        if tokens and ev.get("model"):
            for typ, key in (("input", "prompt"), ("output", "completion"), ("cache_read", "cached")):
                if tokens.get(key):
                    self.observe_hist("gen_ai_client_token_usage",
                                      {"gen_ai_operation_name": "chat", "gen_ai_provider_name": "anthropic",
                                       "gen_ai_request_model": ev["model"], "gen_ai_token_type": typ, "tier": tier},
                                      tokens[key], TOKEN_BUCKETS)
        if ev.get("steps"):
            self.inc("codemap_steps_total", {"tier": tier}, ev["steps"])
        if et == "ask" and ev.get("duration_ms") is not None:
            self.observe_hist("gen_ai_client_operation_duration_seconds", {"tier": tier},
                              ev["duration_ms"] / 1000.0, SECONDS_BUCKETS)
        if et == "feedback":
            if ev.get("rating") is not None:
                self.inc("codemap_feedback_total", {"tier": ev.get("tier") or "none", "rating": str(ev["rating"])})
            for tag in ev.get("tags") or []:
                self.inc("codemap_feedback_tags_total", {"tag": tag})
            self.inc("codemap_feedback_verified_total", {"verified": "true" if ev.get("verified") else "false"})
        if et == "miss":
            repo = (ev.get("path") or "").split("/", 1)[0] or "?"
            self.inc("codemap_miss_total", {"repo": repo})
        if et == "budget_refusal":
            self.inc("codemap_budget_refusals_total", {"user": user})

    # ------------------------------------------------------------------ exposition
    def render(self):
        out = []
        done = set()

        def head(name):
            if name not in done:
                out.append(f"# HELP {name} {self.help.get(name, '')}")
                out.append(f"# TYPE {name} {self.types.get(name, 'untyped')}")
                done.add(name)

        with self._lock:
            for (name, labels), v in sorted(self.counters.items()):
                head(name)
                out.append(f"{name}{_labels(dict(labels))} {_num(v)}")
            for (name, labels), v in sorted(self.gauges.items()):
                head(name)
                out.append(f"{name}{_labels(dict(labels))} {_num(v)}")
            for (name, labels), h in sorted(self.hists.items()):
                head(name)
                base = dict(labels)
                for i, e in enumerate(h["edges"]):
                    out.append(f"{name}_bucket{_labels(dict(base, le=_num(e)))} {h['buckets'][i]}")
                out.append(f"{name}_bucket{_labels(dict(base, le=INF))} {h['buckets'][-1]}")
                out.append(f"{name}_sum{_labels(base)} {_num(h['sum'])}")
                out.append(f"{name}_count{_labels(base)} {h['count']}")
        return "\n".join(out) + "\n"

    def influx_lines(self, measurement_prefix="", ts_ns=None):
        """Counters and gauges as Influx line protocol (running totals: max_over_time on public panels is exact)."""
        import time
        ts = ts_ns or time.time_ns()
        lines = []
        with self._lock:
            for (name, labels), v in sorted(list(self.counters.items()) + list(self.gauges.items())):
                tags = "".join(f",{k}={_tag(v2)}" for k, v2 in labels)
                lines.append(f"{measurement_prefix}{name}{tags} value={_num(v)} {ts}")
        return lines


def _num(v):
    if isinstance(v, float):
        if math.isinf(v):
            return INF
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(v)
    return str(v)


def _tag(v):
    return str(v).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")
