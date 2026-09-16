"""Provision the dashboards to Grafana Cloud and make them public; write public-urls.md; verify one
panel through the public query API; optionally post an annotation.

    python tools/grafana/provision.py --upsert --public [--annotate "prompt v2 promoted"] [--verify]

Needs GRAFANA_CLOUD_SA_TOKEN (dashboards API) in the environment (~/.grafana-cloud.env). The
`${DS_PROMETHEUS}` input is bound to the stack's Prometheus datasource uid. Stdlib only.
"""

import argparse
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.environ.get("GRAFANA_URL", "https://checkitoutapp.grafana.net")
DS_UID = os.environ.get("GRAFANA_PROM_UID", "grafanacloud-prom")
URLS_MD = os.path.join(R, "observability", "grafana", "public-urls.md")


def api(method, path, body=None, token=None, timeout=30):
    token = token or os.environ.get("GRAFANA_CLOUD_SA_TOKEN")
    if not token:
        raise SystemExit("GRAFANA_CLOUD_SA_TOKEN missing")
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # NOSONAR - Grafana Cloud API over https; see sonar-project.properties
            return r.status, json.loads(r.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8") or "null")
        except ValueError:
            return e.code, None


def bind_inputs(doc):
    text = json.dumps(doc).replace("${DS_PROMETHEUS}", DS_UID)
    d = json.loads(text)
    d.pop("__inputs", None)
    d.pop("__requires", None)
    d.pop("id", None)
    return d


def upsert(doc, folder_uid=None):
    body = {"dashboard": bind_inputs(doc), "overwrite": True, "message": "codemap provision.py"}
    if folder_uid:
        body["folderUid"] = folder_uid
    code, res = api("POST", "/api/dashboards/db", body)
    if code != 200:
        raise SystemExit(f"upsert {doc['uid']} failed: {code} {res}")
    return res


def make_public(uid):
    code, res = api("GET", f"/api/dashboards/uid/{uid}/public-dashboards")
    if code == 200 and res and res.get("accessToken"):
        if not res.get("isEnabled"):
            api("PATCH", f"/api/dashboards/uid/{uid}/public-dashboards/{res['uid']}", {"isEnabled": True})
        return res["accessToken"]
    code, res = api("POST", f"/api/dashboards/uid/{uid}/public-dashboards", {"isEnabled": True, "timeSelectionEnabled": True,
                                                                             "annotationsEnabled": True, "share": "public"})
    if code not in (200, 201):
        raise SystemExit(f"public-dashboard for {uid} failed: {code} {res}")
    return res["accessToken"]


def verify_panel(token, panel_id):
    now = int(time.time() * 1000)
    body = {"intervalMs": 60000, "maxDataPoints": 100, "timeRange": {"from": str(now - 14 * 86400 * 1000), "to": str(now), "timezone": "utc"}}
    code, res = api("POST", f"/api/public/dashboards/{token}/panels/{panel_id}/query", body)
    if code != 200 or not res:
        return code, 0
    frames = 0
    for r in (res.get("results") or {}).values():
        for f in r.get("frames", []) or []:
            vals = (f.get("data") or {}).get("values") or []
            frames += 1 if vals and any(vals) else 0
    return code, frames


def annotate(text, tags=("codemap", "promotion")):
    code, res = api("POST", "/api/annotations", {"text": text, "tags": list(tags), "time": int(time.time() * 1000)})
    return code


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--upsert", action="store_true")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--annotate", default=None)
    ap.add_argument("--folder-uid", default=None)
    a = ap.parse_args(argv)
    if a.annotate:
        print("annotation", annotate(a.annotate))
    paths = sorted(glob.glob(os.path.join(R, "observability", "grafana", "codemap-*.json")))
    rows = []
    for p in paths:
        doc = json.load(open(p, encoding="utf-8"))
        uid, title = doc["uid"], doc["title"]
        url = f"{BASE}/d/{uid}"
        if a.upsert:
            res = upsert(doc, a.folder_uid)
            url = BASE + res.get("url", f"/d/{uid}")
            print(f"upserted {uid} v{res.get('version')}")
        public = None
        if a.public:
            tok = make_public(uid)
            public = f"{BASE}/public-dashboards/{tok}"
            print(f"public {uid}: {public}")
            if a.verify:
                first_stat = next((pp["id"] for pp in doc["panels"] if pp.get("type") in ("stat", "timeseries", "bargauge")), None)
                code, frames = verify_panel(tok, first_stat)
                print(f"  verify panel {first_stat}: http {code}, frames with data {frames}")
        rows.append((title, uid, url, public))
    with open(URLS_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write("# CodeMap dashboards — public URLs\n\nProvisioned by `tools/grafana/provision.py --upsert --public` from the JSON beside this file. "
                "Public dashboards take no template variables; counters are shown as `max_over_time` of running totals.\n\n| dashboard | uid | public URL |\n|---|---|---|\n")
        for title, uid, url, public in rows:
            f.write(f"| {title} | `{uid}` | {public or '(not public yet)'} |\n")
    print(f"wrote {URLS_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
