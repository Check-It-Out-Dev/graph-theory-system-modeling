"""The night: six personas hold conversations with CodeMap from a checkout, verify, rate; a seeded
fifth of the conversations run without CodeMap so gains are measured.

    python eval/humans/run_night.py --date 2026-09-17 [--dry-run] [--persona haiku-pm] [--limit 1]
                                    [--haiku-only] [--max-credits 3000] [--seed 7] [--sync-agents]

Env: CODEMAP_URL, CODEMAP_TOKEN, CODEMAP_ADMIN_TOKEN (backlog sync), CODEMAP_REPO_BACKEND /
CODEMAP_REPO_FRONTEND (checkouts the personas work in), GRAFANA_CLOUD_ALLOY_TOKEN (Claude Code OTEL
to Grafana Cloud when present). Every model call goes through app/claude_cli.py (subscription, role
tagged). A rate-limited result ends the night as partial; nothing retries in a loop.
"""

import argparse
import base64
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "app"))
sys.path.insert(0, R)

import claude_cli  # noqa: E402

ROLE_ORDER = ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-5"]
_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
OTLP = "https://otlp-gateway-prod-eu-west-2.grafana.net/otlp"
STACK = "1359921"


# ----------------------------------------------------------------------------- inputs

def load_personas(path=None):
    return json.load(open(path or os.path.join(HERE, "personas.json"), encoding="utf-8"))


def load_bank():
    rows = [json.loads(l) for l in open(os.path.join(R, "eval", "q", "mfq_all.jsonl"), encoding="utf-8") if l.strip()]
    probes = [json.loads(l) for l in open(os.path.join(R, "eval", "q", "probes_offdist.jsonl"), encoding="utf-8") if l.strip()]
    return rows, probes


def slice_for(persona, bank, probes):
    sl = persona["slice"]
    pool = [r for r in bank if r.get("role") in sl["roles"] or r.get("archetype") in sl["archetypes"]]
    pool += [dict(p, role="probe", archetype=p["kind"]) for p in probes if p["kind"] in sl["probe_kinds"]]
    return pool


# ----------------------------------------------------------------------------- plan

def plan(date, cfg, bank, probes, seed=None, only=None, limit=None, haiku_only=False):
    """Deterministic list of conversations for the night."""
    convs = []
    for pid in cfg["order"]:
        p = next(x for x in cfg["personas"] if x["id"] == pid)
        if only and p["id"] != only:
            continue
        if haiku_only and not p["model"].startswith("claude-haiku"):
            continue
        rnd = random.Random(f"{date}:{seed or 0}:{p['id']}")
        pool = slice_for(p, bank, probes)
        if not pool:
            continue
        n = p["conversations_per_night"] if limit is None else min(limit, p["conversations_per_night"])
        seeds = rnd.sample(pool, min(n, len(pool)))
        lo, hi = p["turns_per_conversation"]
        for i, s in enumerate(seeds):
            turns = rnd.randint(lo, hi)
            fus = rnd.sample(p["follow_ups"], min(turns - 1, len(p["follow_ups"])))
            mode = "baseline" if (rnd.random() < cfg.get("baseline_share", 0.2) and i > 0) else "codemap"
            convs.append({"persona": p["id"], "model": p["model"], "repo": p["repo"], "role_file": p["role_file"],
                          "mode": mode, "seed_id": s["id"], "seed_q": s["q"], "kind": s.get("archetype"),
                          "expect": s.get("expect", "answer"), "turns": turns, "follow_ups": fus,
                          "credits_per_night": p["credits_per_night"], "n": i + 1, "of": len(seeds)})
    # a gain is a pair: the same persona, the same seed, once without CodeMap and once with it. A baseline
    # without its partner is a number nobody can gate, so the partner is scheduled right after it.
    have = {(c["persona"], c["seed_id"]) for c in convs if c["mode"] == "codemap"}
    for c in [c for c in convs if c["mode"] == "baseline"]:
        if (c["persona"], c["seed_id"]) not in have:
            partner = dict(c, mode="codemap", n=c["n"], of=c["of"], paired_with="baseline")
            convs.insert(convs.index(c) + 1, partner)
            have.add((c["persona"], c["seed_id"]))
    return convs


def conversation_prompt(c):
    fus = "\n".join(f"  - {f}" for f in c["follow_ups"])
    if c["mode"] == "baseline":
        return (f"BASELINE SESSION (no CodeMap tonight; find the answer in this checkout with Read, Grep and Glob only).\n"
                f"Conversation {c['n']} of {c['of']}. The question, in your own words: {c['seed_q']}\n"
                f"Follow-up intents you would have asked:\n{fus}\n"
                f"Answer each as well as you can from the files, cite the files you opened, then finish with the json block "
                f"(use \"baseline\" as context_id and request_id, rate your own confidence 1-5 as rating, verified=true when you "
                f"opened the file, and set would_have_found_alone and minutes_saved_estimate honestly).")
    return (f"Conversation {c['n']} of {c['of']} tonight. Seed question (rephrase it the way you would say it): {c['seed_q']}\n"
            f"Plan {c['turns']} turns in ONE context_id: the seed, then follow-ups drawn from:\n{fus}\n"
            f"Verify at least one pointer per answer in this checkout before rating; call codemap_feedback after every answer; "
            f"report misses; finish with the json block.")


def mcp_config(c, url, token):
    if c["mode"] == "baseline":
        return {"mcpServers": {}}
    return {"mcpServers": {"codemap": {"type": "http", "url": url.rstrip("/") + "/mcp",
                                       "headers": {"Authorization": f"Bearer {token}", "X-CodeMap-User": c["persona"]}}}}


def system_file(c, tmpdir):
    role = open(os.path.join(HERE, c["role_file"]), encoding="utf-8").read()
    common = open(os.path.join(HERE, "roles", "_common.md"), encoding="utf-8").read()
    path = os.path.join(tmpdir, f"role-{c['persona']}.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(role.rstrip() + "\n\n" + common)
    return path


def otel_env(role, persona, night, env=None):
    env = env if env is not None else os.environ
    tok = env.get("GRAFANA_CLOUD_ALLOY_TOKEN") or env.get("GRAFANA_ALLOY_TOKEN")
    if not tok:
        return {}
    auth = base64.b64encode(f"{STACK}:{tok}".encode()).decode()
    return {"CLAUDE_CODE_ENABLE_TELEMETRY": "1", "OTEL_METRICS_EXPORTER": "otlp", "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf", "OTEL_EXPORTER_OTLP_ENDPOINT": env.get("OTEL_EXPORTER_OTLP_ENDPOINT", OTLP),
            "OTEL_EXPORTER_OTLP_HEADERS": f"Authorization=Basic {auth}",
            "OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE": "cumulative", "OTEL_METRICS_INCLUDE_ACCOUNT_UUID": "false",
            "OTEL_METRIC_EXPORT_INTERVAL": "10000", "OTEL_LOGS_EXPORT_INTERVAL": "5000",
            "OTEL_RESOURCE_ATTRIBUTES": f"service.name=codemap,role={role},persona={persona},night={night}"}


def parse_report(text):
    m = None
    for m in _JSON_BLOCK.finditer(text or ""):
        pass
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


# ----------------------------------------------------------------------------- server calls

def budget(url, token, user, http=None):
    if http is not None:
        return http("GET", f"{url}/budget?user={user}", None)
    req = urllib.request.Request(f"{url.rstrip('/')}/budget?user={user}", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:  # NOSONAR - our server over https; see sonar-project.properties
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return json.load(e)
        except ValueError:
            return {"exhausted": e.code == 429, "spent": None, "remaining": None}
    except (urllib.error.URLError, OSError):
        return None


def sync_backlog(url, token, admin, dest, http=None):
    if not admin:
        return 0
    try:
        if http is not None:
            rows = http("GET", f"{url}/admin/backlog", None)
        else:
            req = urllib.request.Request(f"{url.rstrip('/')}/admin/backlog",
                                         headers={"Authorization": f"Bearer {token}", "X-CodeMap-Admin": admin})
            with urllib.request.urlopen(req, timeout=30) as r:  # NOSONAR - our server over https; see sonar-project.properties
                rows = json.load(r).get("rows", [])
    except (urllib.error.URLError, OSError, ValueError):
        return 0
    have = set()
    if os.path.exists(dest):
        for line in open(dest, encoding="utf-8"):
            have.add(line.strip())
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    n = 0
    with open(dest, "a", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if line not in have:
                f.write(line + "\n")
                have.add(line)
                n += 1
    return n


# ----------------------------------------------------------------------------- night

def run(args, runner=None, http=None, env=None):
    env = env if env is not None else os.environ
    cfg = load_personas()
    bank, probes = load_bank()
    if getattr(args, "baseline_share", None) is not None:
        cfg = dict(cfg, baseline_share=args.baseline_share)
    convs = plan(args.date, cfg, bank, probes, args.seed, args.persona, args.limit, args.haiku_only)
    url = env.get("CODEMAP_URL", "http://127.0.0.1:7345")
    token = env.get("CODEMAP_TOKEN", "")
    repos = {"backend": env.get("CODEMAP_REPO_BACKEND", os.path.expanduser("~/IdeaProjects/checkitout-backend")),
             "frontend": env.get("CODEMAP_REPO_FRONTEND", os.path.expanduser("~/IdeaProjects/checkitout-frontend"))}
    out_dir = os.path.join(HERE, "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.date}.jsonl")
    if args.dry_run:
        for c in convs:
            print(f"{c['persona']:16} {c['mode']:8} {c['seed_id']:5} turns={c['turns']} {c['seed_q'][:70]}")
        print(f"{len(convs)} conversations, {sum(1 for c in convs if c['mode'] == 'baseline')} baseline")
        return {"planned": len(convs), "dry_run": True}
    import tempfile
    tmp = tempfile.mkdtemp(prefix="codemap-night-")
    summary = {"night": args.date, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "conversations": 0,
               "partial": False, "stopped_reason": None, "by_persona": {}, "credits_cap": args.max_credits}
    spent_start = {}
    for c in convs:
        pid = c["persona"]
        bp = summary["by_persona"].setdefault(pid, {"conversations": 0, "baseline": 0, "turns": 0, "errors": 0,
                                                      "usage": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
                                                      "credits_spent": 0.0, "ratings": [], "misses": 0})
        if c["mode"] == "codemap":
            b = budget(url, token, pid, http)
            if b is None:
                summary["partial"], summary["stopped_reason"] = True, "server unreachable"
                break
            spent_start.setdefault(pid, b.get("spent") or 0.0)
            if b.get("exhausted"):
                bp["errors"] += 1
                _append(out_path, {"night": args.date, "persona": pid, "mode": c["mode"], "seed_id": c["seed_id"],
                                   "skipped": "budget_exhausted"})
                continue
            total_spent = sum((budget(url, token, p, http) or {}).get("spent") or 0 for p in spent_start) - sum(spent_start.values())
            if args.max_credits and total_spent >= args.max_credits:
                summary["partial"], summary["stopped_reason"] = True, f"global credit cap {args.max_credits}"
                break
        role = "baseline" if c["mode"] == "baseline" else "persona"
        cwd = repos.get(c["repo"]) if os.path.isdir(repos.get(c["repo"], "")) else None
        res = claude_cli.run(conversation_prompt(c), c["model"], role=role, system_file=system_file(c, tmp),
                             mcp_config=mcp_config(c, url, token),
                             allowed_tools=("mcp__codemap__*", "Read", "Grep", "Glob") if role == "persona" else ("Read", "Grep", "Glob"),
                             max_turns=args.max_turns, persona=pid, timeout=args.timeout, cwd=cwd,
                             env_extra=otel_env(role, pid, args.date, env), runner=runner, append_system=True)
        report = parse_report(res.get("text"))
        row = {"night": args.date, "persona": pid, "model": c["model"], "mode": c["mode"], "seed_id": c["seed_id"],
               "seed_q": c["seed_q"], "kind": c["kind"], "expect": c["expect"], "planned_turns": c["turns"],
               "session_claude": res.get("session_id"), "num_turns": res.get("num_turns"), "usage": res.get("usage"),
               "cost_est": res.get("cost_usd"), "duration_ms": res.get("duration_ms"), "is_error": res.get("is_error"),
               "rate_limited": res.get("rate_limited"), "error": res.get("error"), "report": report,
               "result_tail": (res.get("text") or "")[-1500:]}
        _append(out_path, row)
        bp["conversations"] += 1
        bp["baseline"] += c["mode"] == "baseline"
        bp["turns"] += res.get("num_turns") or 0
        u = res.get("usage") or {}
        bp["usage"]["input"] += u.get("input_tokens", 0)
        bp["usage"]["output"] += u.get("output_tokens", 0)
        bp["usage"]["cache_read"] += u.get("cache_read_input_tokens", 0)
        bp["usage"]["cache_creation"] += u.get("cache_creation_input_tokens", 0)
        if report:
            for conv in report.get("conversations", []):
                for t in conv.get("turns", []):
                    if isinstance(t.get("rating"), int):
                        bp["ratings"].append(t["rating"])
            bp["misses"] += len(report.get("misses") or [])
        summary["conversations"] += 1
        if res.get("is_error"):
            bp["errors"] += 1
            if res.get("rate_limited"):
                summary["partial"], summary["stopped_reason"] = True, "rate limited"
                break
    for pid in summary["by_persona"]:
        b = budget(url, token, pid, http) or {}
        summary["by_persona"][pid]["credits_spent"] = round((b.get("spent") or 0) - spent_start.get(pid, 0), 4)
        r = summary["by_persona"][pid]["ratings"]
        summary["by_persona"][pid]["rating_mean"] = round(sum(r) / len(r), 3) if r else None
    summary["backlog_synced"] = sync_backlog(url, token, env.get("CODEMAP_ADMIN_TOKEN"),
                                             os.path.join(R, "graph", "delta", "backlog.jsonl"), http)
    summary["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(out_dir, f"{args.date}.summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    return summary


def _append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sync_agents():
    """Ship each role as applications/CodeMap/.claude/agents/<id>.md so the owner can run a persona by hand."""
    cfg = load_personas()
    out = os.path.join(R, ".claude", "agents")
    os.makedirs(out, exist_ok=True)
    common = open(os.path.join(HERE, "roles", "_common.md"), encoding="utf-8").read()
    for p in cfg["personas"]:
        role = open(os.path.join(HERE, p["role_file"]), encoding="utf-8").read()
        first = role.splitlines()[0].lstrip("# ").strip()
        model = {"claude-opus-5": "opus", "claude-sonnet-5": "sonnet"}.get(p["model"], "haiku")
        body = (f"---\nname: {p['id']}\ndescription: CodeMap synthetic user — {first}. Needs the codemap MCP.\nmodel: {model}\n---\n\n"
                + role.rstrip() + "\n\n" + common)
        with open(os.path.join(out, f"{p['id']}.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
    return len(cfg["personas"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d", time.gmtime()))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--persona", default=None)
    ap.add_argument("--limit", type=int, default=None, help="conversations per persona")
    ap.add_argument("--haiku-only", action="store_true")
    ap.add_argument("--max-credits", type=float, default=3000)
    ap.add_argument("--baseline-share", type=float, default=None, help="override personas.json baseline_share (the pair campaign uses 0.5)")
    ap.add_argument("--max-turns", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sync-agents", action="store_true")
    a = ap.parse_args(argv)
    if a.sync_agents:
        print(f"synced {sync_agents()} agent files")
        return 0
    s = run(a)
    print(json.dumps(s, indent=1, sort_keys=True)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
