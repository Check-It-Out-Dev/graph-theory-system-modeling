"""The night's events from the served instance, byte-for-byte, into eval/judge/runs/events-<date>.jsonl.

    python eval/judge/fetch_events.py --date D [--since ISO] [--limit 5000]

The window starts where the personas started (`started` in eval/humans/runs/<date>.summary.json):
the night label may sit ahead of the box clock, so its midnight is no window at all (INCIDENTS
2026-09-19). Events are written as the server sent them - never round-tripped through a shell's JSON
objects, which rewrite ISO timestamps. Needs CODEMAP_URL, CODEMAP_TOKEN, CODEMAP_ADMIN_TOKEN.
Exit 2 when the window holds no events (the night script stops instead of judging nothing).
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))


def window_start(date, runs_dir=None, since=None):
    """-> the ISO instant the events window opens at: --since, else the summary's `started`, else the label's midnight."""
    if since:
        return since
    path = os.path.join(runs_dir or os.path.join(R, "eval", "humans", "runs"), f"{date}.summary.json")
    if os.path.exists(path):
        try:
            started = json.load(open(path, encoding="utf-8")).get("started")
            if started:
                return started
        except (OSError, ValueError):
            pass
    return f"{date}T00:00:00Z"


def fetch(url, token, admin, since, limit=5000, opener=None):
    q = urllib.parse.urlencode({"since": since, "limit": limit})
    req = urllib.request.Request(f"{url.rstrip('/')}/admin/events?{q}",
                                 headers={"Authorization": f"Bearer {token}", "X-CodeMap-Admin": admin})
    if opener is not None:
        return opener(req)
    with urllib.request.urlopen(req, timeout=60) as r:  # NOSONAR - our own server over https
        return json.load(r)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--since", default=None)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    since = window_start(a.date, since=a.since)
    doc = fetch(os.environ["CODEMAP_URL"], os.environ["CODEMAP_TOKEN"], os.environ["CODEMAP_ADMIN_TOKEN"], since, a.limit)
    events = doc.get("events") or []
    out = a.out or os.path.join(HERE, "runs", f"events-{a.date}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"events fetched: {len(events)} since {since} -> {out}")
    return 0 if events else 2


if __name__ == "__main__":
    sys.exit(main())
