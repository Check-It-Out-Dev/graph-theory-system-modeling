"""Credits: a currency-free price of an answer, and the daily ledger that enforces budgets.

credits = ((prompt - cached) * in + cached * in * m_read + cache_creation * in * m_create
           + completion * out) / 1000 + flat, rounded to 4 dp. The ledger keys spend on the
event's UTC date and is rebuilt from the events file at boot, so a restart forgets nothing and
a budget is a fact about the artifact, not about process memory.
"""

import json
import os
import threading
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path=None):
    path = path or os.environ.get("CODEMAP_CREDITS") or os.path.join(HERE, "credits.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class RateCard:
    def __init__(self, doc=None):
        self.doc = doc or load()
        self.per_1k = self.doc["per_1k"]
        self.m_read = float(self.doc.get("cache_read_multiplier", 0.1))
        self.m_create = float(self.doc.get("cache_creation_multiplier", 1.25))
        self.flat = self.doc.get("flat", {})

    def compute(self, tier, tokens=None, flat=None):
        rate = self.per_1k.get(tier) or {"in": 0.0, "out": 0.0}
        t = tokens or {}
        prompt = float(t.get("prompt", 0) or 0)
        cached = float(t.get("cached", 0) or 0)
        creation = float(t.get("cache_creation", 0) or 0)
        completion = float(t.get("completion", 0) or 0)
        fresh = max(0.0, prompt - cached)
        c = (fresh * rate["in"] + cached * rate["in"] * self.m_read
             + creation * rate["in"] * self.m_create + completion * rate["out"]) / 1000.0
        if flat:
            c += float(self.flat.get(flat, 0.0))
        return round(c, 4)

    def table(self):
        """Rows for the dashboard's rate card (markdown), no currency."""
        rows = ["| tier | credits per 1k input | per 1k output |", "|---|---|---|"]
        for tier, r in self.per_1k.items():
            rows.append(f"| {tier} | {r['in']} | {r['out']} |")
        rows.append(f"| cache read | x{self.m_read} of input | |")
        rows.append(f"| cache creation | x{self.m_create} of input | |")
        for k, v in self.flat.items():
            rows.append(f"| {k} (flat) | {v} | |")
        return "\n".join(rows)


def utc_date(ts=None):
    if ts:
        return ts[:10]
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def seconds_to_utc_midnight(now=None):
    now = now or datetime.now(timezone.utc)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((nxt - now).total_seconds())


class Ledger:
    """spent[(user, utc_date)] += credits; refusals counted; rebuilt from events at boot."""

    def __init__(self):
        self._spent = {}
        self._requests = {}
        self._lock = threading.Lock()

    def replay(self, events):
        n = 0
        for ev in events:
            self.observe(ev)
            n += 1
        return n

    def observe(self, ev):
        c = float(ev.get("credits") or 0)
        key = (ev.get("user"), utc_date(ev.get("ts")))
        with self._lock:
            self._spent[key] = self._spent.get(key, 0.0) + c
            if ev.get("event_type") == "ask":
                self._requests[key] = self._requests.get(key, 0) + 1

    def spent(self, user, date=None):
        return round(self._spent.get((user, date or utc_date()), 0.0), 4)

    def requests(self, user, date=None):
        return self._requests.get((user, date or utc_date()), 0)

    def state(self, user_doc, date=None):
        budget = float(user_doc["daily_credit_budget"])
        spent = self.spent(user_doc["id"], date)
        return {"user": user_doc["id"], "date": date or utc_date(), "budget": budget, "spent": spent,
                "remaining": round(max(0.0, budget - spent), 4), "requests": self.requests(user_doc["id"], date),
                "exhausted": spent >= budget, "resets_in_s": seconds_to_utc_midnight()}
