# CodeMap Remote — the served MCP

The graph a team points Claude Code at. One stdlib Python process (plus `real-ladybug` for the pack)
answers seven MCP tools over Streamable HTTP; the navigator tier is Claude Sonnet driving the engine
through a loopback MCP, run as the `claude` CLI on the owner's subscription. Every call writes one
event line; credits, budgets, metrics and dashboards are views of that file.

Live instance: `https://codemap.checkitout.app/mcp` on the demo VPS. Design: `docs/07-ai-quality-governance.md`
(§ Revision 2026-09-16 and the D-R decision log).

## Connect from Claude Code

```
claude mcp add --transport http codemap https://codemap.checkitout.app/mcp \
  --header "Authorization: Bearer ${CODEMAP_TOKEN}" --header "X-CodeMap-User: ${CODEMAP_USER}"
```

`CODEMAP_USER` must be a name from `users.json` (the enum: `owner`, six personas, `anonymous`, the system
roles). Never name the variable like a credential (`*_API_KEY`, `ANTHROPIC_*`): Claude Code reads those as
empty in MCP headers.

| tool | what | credits |
|---|---|---|
| `codemap_ask(q, context_id?, tier?)` | the navigator answers with prose + pointers; reuse `context_id` for follow-ups (they resume the Claude session) | per the rate card (`nav-sonnet` ≈ 20–30 per answer; `deep` = Opus, 5×); FAQ hits are free |
| `codemap_step(dsl)` | one CMDSL verb against the graph | free |
| `codemap_open(name)` | exact entity name → pointer (repo, relative path, subsystem, clue) | free |
| `codemap_search(text, k?)` | Qwen3 embedding + reranker over entity sockets | 2 + query tokens |
| `codemap_feedback(request_id, rating\|vote, tags?, comment?, verified?)` | rate an answer; a rating above 3 needs `verified=true` | free |
| `codemap_miss(path, why?)` | a file the graph did not know → the saturation backlog | free |
| `codemap_status()` | versions, your budget, queue depth | free |

Pointers, never content: the server does not read or return file bodies. Open the pointer in your own
checkout, then rate.

## Security, in one paragraph

Security is designed to serve a small team, even behind a firewall — that is why all users can switch
names: identity is a header from an enum and one shared token. This deployment exists to show how to
measure prompt quality in an AI system and how to observe it. A full version needs a separate account for
each person (OAuth 2.1 as the MCP specification describes); that is out of scope here. The hosted instance
runs on the owner's own Claude subscription through the Claude Code CLI; nobody else's token is involved.
In normal use a team of ten runs the server locally (below) against their own subscription or API key.

## REST beside the MCP

`GET /healthz` (no auth) · `GET /status` · `GET /users` · `GET /budget?user=` (429 when exhausted) ·
`GET /metrics` (Prometheus text; OTel GenAI semconv histograms + `codemap_*`) · `POST /feedback` (the
tool's REST twin) · `POST /admin/reload` (bearer + `X-CodeMap-Admin`: fetch the latest pack Release,
swap the engine, rebuild the active prompt and the search index).

## Run it locally (the ten-person-team story)

```
python applications/CodeMap/tools/pack/fetch_pack.py --latest     # the graph, sha256-verified
cd applications/CodeMap && python -m remote.server                  # 127.0.0.1:7345, no token needed on loopback
claude mcp add --transport http codemap http://127.0.0.1:7345/mcp --header "X-CodeMap-User: owner"
```

The navigator runs when `claude` is on PATH (your login or `CLAUDE_CODE_OAUTH_TOKEN`); without it the
server still answers FAQ hits and engine steps. `CODEMAP_NAVIGATOR=off` / `CODEMAP_SEARCH=off` switch the
paid tiers off; `CODEMAP_NAV_MODEL` picks the model.

## Deploy (the VPS)

One path, from a box or from `deploy-codemap.yml` (dispatch-only, `codemap-vps` environment):

```
bash applications/CodeMap/tools/deploy.sh --host gvps            # ship HEAD, build there, roll, verify
```

Provisioned once by hand on the host, never in git: `/opt/codemap/.env` (`CODEMAP_TOKEN`, `CODEMAP_ADMIN_TOKEN`,
`CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`, the Grafana Cloud push credentials, Modal URLs), the
Let's Encrypt certificate (`certbot certonly --webroot -w /var/www/le -d codemap.checkitout.app`), the DNS
record (DNS-only at Cloudflare: a navigator turn can outlive the proxy's timeout). The container is capped
at 1.2 GB with two concurrent navigator turns; the pack, the telemetry and the CLI's sessions live on volumes.

## Telemetry

`telemetry/events.jsonl` is the artifact of record (schema in `telemetry.py`). `GET /metrics` is a view of it;
the pusher sends running totals as Influx lines and every event line to Loki (Grafana Cloud) when the
`GRAFANA_*` variables are set. `telemetry/fixtures/events.sample.jsonl` (200 rows, seed 42) is what CI replays.

## Tests

```
cd applications/CodeMap && PYTHONUTF8=1 python -m unittest discover -s remote/tests -t .
```

No test calls a model or the network: the CLI is a fake subprocess, Modal is a fake transport that
redirects once, the pack-backed checks skip when the pack is absent (CI fetches it from the Release).
