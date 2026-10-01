"""Telemetry of a campaign, into the estate's Grafana Cloud stack (the same one the served MCP and the persona nights use).

    enable(label)          Claude Code's own OpenTelemetry for every session of the campaign (coder, judge, reflector,
                           preflight), labelled service.name=codemap-put, campaign=<label>, role=<role>, prompt_version
    push(summary)          the campaign's gauges as Influx lines: put_score, put_rule_rate{rule}, put_delta,
                           put_runs, put_budget_exhausted, put_holdout_gain (certification)

Credentials: the environment, else ~/.grafana-cloud.env on the box (GRAFANA_CLOUD_ALLOY_TOKEN), the file the persona
nights already read. Nothing is pushed without a token, and a failed push never fails a campaign: the committed
artifacts are the record, the dashboard is a view of them.
"""

import base64
import os
import time

OTLP = "https://otlp-gateway-prod-eu-west-2.grafana.net/otlp"
STACK = "1359921"
ENV_FILE = os.path.join(os.path.expanduser("~"), ".grafana-cloud.env")


def _load_env_file():
    if os.environ.get("GRAFANA_CLOUD_ALLOY_TOKEN") or os.environ.get("GRAFANA_ALLOY_TOKEN") or not os.path.exists(ENV_FILE):
        return
    with open(ENV_FILE, encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                if k.startswith("GRAFANA_") and v:
                    os.environ.setdefault(k, v.strip().strip('"'))


def token():
    _load_env_file()
    return os.environ.get("GRAFANA_CLOUD_ALLOY_TOKEN") or os.environ.get("GRAFANA_ALLOY_TOKEN")


def enable(label):
    """Set the OTLP exporter for every child claude session. -> True when telemetry is on."""
    tok = token()
    if not tok:
        return False
    auth = base64.b64encode(f"{STACK}:{tok}".encode()).decode()
    os.environ.update({
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1", "OTEL_METRICS_EXPORTER": "otlp", "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf", "OTEL_EXPORTER_OTLP_ENDPOINT": os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", OTLP),
        "OTEL_EXPORTER_OTLP_HEADERS": f"Authorization=Basic {auth}",
        "OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE": "cumulative", "OTEL_METRICS_INCLUDE_ACCOUNT_UUID": "false",
        "OTEL_METRIC_EXPORT_INTERVAL": "10000", "OTEL_LOGS_EXPORT_INTERVAL": "5000",
        "OTEL_RESOURCE_ATTRIBUTES": f"service.name=codemap-put,campaign={label}",
    })
    return True


def lines(summary, ts=None):
    ts = ts or time.time_ns()
    tags = f"campaign={summary['label']},mode={summary.get('mode')},prompt={summary.get('prompt_version', '').replace('@', '_')}"
    out = []

    def emit(name, value, extra=""):
        if value is not None:
            out.append(f"{name},{tags}{extra} value={float(value)} {ts}")
    emit("put_score", summary.get("score"))
    emit("put_runs", summary.get("runs"))
    emit("put_delta", (summary.get("noise") or {}).get("delta"))
    for rid, r in (summary.get("rules") or {}).items():
        if r.get("n"):
            emit("put_rule_rate", r["rate"], f",rule={rid}")
            emit("put_rule_wilson_lo", r["wilson"][0], f",rule={rid}")
    emit("put_budget_exhausted", summary.get("budget_exhausted"))
    emit("put_output_tokens", summary.get("output_tokens"))
    v = summary.get("verdict") or {}
    if v:
        emit("put_holdout_gain", v["gain_holdout"]["mean"])
        emit("put_holdout_gain_ci_lo", v["gain_holdout"]["ci"][0])
        emit("put_holdout_gain_ci_hi", v["gain_holdout"]["ci"][1])
    return out


def push(summary):
    """-> number of lines pushed (0 when there is no token or the push failed)."""
    if not token():
        return 0
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from remote import telemetry_push
        p = telemetry_push.Pusher(None)
        body = lines(summary)
        if p.influx_url and p.influx_auth and body:
            telemetry_push._post(p.influx_url, "\n".join(body).encode("utf-8"), "text/plain; charset=utf-8", p.influx_auth)
            return len(body)
    except Exception as ex:                                       # the artifacts are the record; the dashboard is a view
        print(f"telemetry push failed: {type(ex).__name__}: {ex}")
    return 0
