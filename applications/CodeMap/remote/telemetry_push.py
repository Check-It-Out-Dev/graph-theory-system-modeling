"""Push the metrics view to Grafana Cloud (Influx line protocol) and the events to Loki (JSON push).

Stdlib only, so the server keeps its one dependency. Counters go out as running totals every
PUSH_INTERVAL seconds and at shutdown, which is what makes `max_over_time(x[$__range])` exact on
public panels; each event line goes to Loki once with low-cardinality labels. Configured by
environment; silent (and counted) when unset or failing — telemetry never fails an answer.

  GRAFANA_INFLUX_URL   https://influx-prod-24-prod-eu-west-2.grafana.net/api/v1/push/influx/write
  GRAFANA_PROM_USER    2644077            GRAFANA_ALLOY_TOKEN   <metrics:write token>
  GRAFANA_LOKI_URL     https://logs-prod-012.grafana.net/loki/api/v1/push
  GRAFANA_LOKI_USER    1317715            GRAFANA_LOKI_TOKEN    <logs:write token>
"""

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request

PUSH_INTERVAL = int(os.environ.get("CODEMAP_PUSH_INTERVAL", "15"))
LOKI_LABEL_KEYS = ("event_type", "tier", "user", "terminal")


def _basic(user, token):
    return "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()


def _post(url, data, content_type, auth, timeout=10, http=None):
    if http is not None:  # tests
        return http(url, data, content_type, auth)
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": content_type, "Authorization": auth})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # NOSONAR - configured https endpoint; see sonar-project.properties
        return r.status


class Pusher:
    def __init__(self, registry, env=None, http=None):
        env = dict(env if env is not None else os.environ)
        # the estate's credential file names the tokens GRAFANA_CLOUD_*; accept both spellings
        for short, long in (("GRAFANA_ALLOY_TOKEN", "GRAFANA_CLOUD_ALLOY_TOKEN"), ("GRAFANA_LOKI_TOKEN", "GRAFANA_CLOUD_LOKI_TOKEN"),
                            ("GRAFANA_PROM_USER", "GRAFANA_CLOUD_PROM_USER"), ("GRAFANA_LOKI_USER", "GRAFANA_CLOUD_LOKI_USER")):
            if not env.get(short) and env.get(long):
                env[short] = env[long]
        self.registry = registry
        self.http = http
        self.influx_url = env.get("GRAFANA_INFLUX_URL")
        self.influx_auth = _basic(env.get("GRAFANA_PROM_USER", ""), env.get("GRAFANA_ALLOY_TOKEN", "")) \
            if env.get("GRAFANA_ALLOY_TOKEN") else None
        self.loki_url = env.get("GRAFANA_LOKI_URL")
        self.loki_auth = _basic(env.get("GRAFANA_LOKI_USER", ""), env.get("GRAFANA_LOKI_TOKEN") or env.get("GRAFANA_ALLOY_TOKEN", "")) \
            if (env.get("GRAFANA_LOKI_TOKEN") or env.get("GRAFANA_ALLOY_TOKEN")) else None
        self.service = env.get("CODEMAP_SERVICE_NAME", "codemap")
        self._pending = []
        self._lock = threading.Lock()
        self.stats = {"influx_ok": 0, "influx_fail": 0, "loki_ok": 0, "loki_fail": 0, "events_sent": 0}
        self._stop = threading.Event()
        self._thread = None

    def enabled(self):
        return bool((self.influx_url and self.influx_auth) or (self.loki_url and self.loki_auth))

    # ------------------------------------------------------------------ events
    def on_event(self, ev):
        if not (self.loki_url and self.loki_auth):
            return
        with self._lock:
            self._pending.append(ev)
            if len(self._pending) > 5000:
                del self._pending[:1000]

    def loki_payload(self, events):
        streams = {}
        for ev in events:
            labels = {"service": self.service}
            for k in LOKI_LABEL_KEYS:
                if ev.get(k):
                    labels[k] = str(ev[k])
            key = tuple(sorted(labels.items()))
            ns = _ts_ns(ev.get("ts"))
            streams.setdefault(key, []).append([str(ns), json.dumps(ev, ensure_ascii=False, sort_keys=True)])
        return {"streams": [{"stream": dict(k), "values": v} for k, v in streams.items()]}

    # ------------------------------------------------------------------ one push
    def push_once(self):
        if self.influx_url and self.influx_auth:
            lines = self.registry.influx_lines()
            if lines:
                try:
                    _post(self.influx_url, "\n".join(lines).encode("utf-8"), "text/plain; charset=utf-8",
                          self.influx_auth, http=self.http)
                    self.stats["influx_ok"] += 1
                except (urllib.error.URLError, OSError, ValueError):
                    self.stats["influx_fail"] += 1
        if self.loki_url and self.loki_auth:
            with self._lock:
                batch, self._pending = self._pending, []
            if batch:
                try:
                    _post(self.loki_url, json.dumps(self.loki_payload(batch)).encode("utf-8"), "application/json",
                          self.loki_auth, http=self.http)
                    self.stats["loki_ok"] += 1
                    self.stats["events_sent"] += len(batch)
                except (urllib.error.URLError, OSError, ValueError):
                    self.stats["loki_fail"] += 1
                    with self._lock:
                        self._pending = batch[-2000:] + self._pending
        return dict(self.stats)

    # ------------------------------------------------------------------ thread
    def start(self):
        if not self.enabled() or self._thread:
            return False
        self._thread = threading.Thread(target=self._loop, name="codemap-push", daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        while not self._stop.wait(PUSH_INTERVAL):
            self.push_once()

    def stop(self):
        self._stop.set()
        if self.enabled():
            self.push_once()


def _ts_ns(iso):
    try:
        from datetime import datetime
        dt = datetime.fromisoformat((iso or "").replace("Z", "+00:00"))
        return int(dt.timestamp() * 1e9)
    except (ValueError, TypeError):
        return time.time_ns()
