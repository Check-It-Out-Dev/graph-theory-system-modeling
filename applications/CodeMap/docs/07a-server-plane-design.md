# CodeMap governance loop — SERVER PLANE design (read-only, 2026-09-14)

Root `R = applications/CodeMap`. All new product code stdlib-only under `R/app/`; MCP under `R/mcp/`.

## 0. Constraints the code fights, and the smallest change

| Constraint | Reality | Smallest change |
|---|---|---|
| "telemetry hook in `run_loop` once" | THREE loops: `loop_runner.run_loop` (4B), `big_tier.run_big_loop` (80B), `api_tier.navigate` (API) | One `telemetry.Meter` object threaded into each loop's `chat()`; the EVENT is emitted in exactly one place, `server.py` (new `handle_ask/handle_step/handle_escalate`). |
| llama-server "launched in codemap.py" | codemap.py never launches it; sites are `model_client.NavigatorModel._ensure` (4B :7348), `big_tier.BigNavigator._ensure` (80B :7352), `loop_runner.main` (eval) | Flags go into `_ensure` ×2; eval launcher stays `-np 1`. |
| `-c 4096` + `-np N` | llama-server splits ctx per slot → 2048/slot, loop overflows | `-c 4096*N`, `N = int(env CODEMAP_SLOTS, 2)`. |
| `rung_cpu.chat` returns `str`; tests mock it with 3-arg lambdas | signature change breaks 2 mocks in `test_app.py` (+2 in big tests) | `chat(port, user, max_tokens, id_slot=None, meter=None)`; mocks become `lambda port, user, mt, **kw:` (4 one-line edits). |
| `H.do_POST` is a monolith → endpoints untestable without a socket | | Extract pure `handle_*(body, headers) -> (code, obj)`; `do_POST` dispatches. |
| repo `.gitignore` ignores `*.json` | `users.json`, `credits.json`, `mcp.json` vanish | negate lines `!applications/CodeMap/app/*.json`, `!applications/CodeMap/mcp/mcp.json`; add `applications/CodeMap/telemetry/*.jsonl` + `!…/telemetry/fixtures/*.jsonl`. |
| `api_tier.navigate` drops `cache_read/creation_input_tokens` | | sum all four usage keys per call into the Meter. |
| `api_tier._log` → `eval/q/api_answers.jsonl` | second record | keep one release, D-log marks it deprecated; event is the record. |

## 1. Users + contexts

- `R/app/users.json` `{schema:1, users:[{id, display_name, role_prompt_ref, model_hint, daily_credit_budget}]}`; `id` regex `^[a-z0-9][a-z0-9-]{1,31}$` (Prometheus-label safe). Ships with `anonymous` (budget 200) so the UI works untouched.
- `R/app/users.py`: `load() -> dict[id, User]`, `resolve(body, headers) -> User | None` (body `user` wins, header `X-CodeMap-User` fallback, default `anonymous`).
- **Unknown user → 400** `{error:"unknown_user", known:[...]}`. Why: personas are the governance dimension; a typo must not mint a persona, and label cardinality must stay bounded. Registration = a commit to `users.json`.
- `R/app/contexts.py`: in-memory LRU (cap 256) `Context{context_id, user, created_at, slot, turns:[{request_id, ts, q, tier, terminal, credits}]}`; `get_or_create(context_id|None, user)`; `slot = zlib.crc32(context_id) % SLOTS`; other user's context → 409. History is NOT injected into the 4B prompt (FREEZE-v1 / D2 train-serve law) — it is record + slot affinity; big/API tiers may consume it later (out of scope).
- Endpoints: `GET /users` → `{users:[{id, display_name, model_hint, daily_credit_budget, spent_today}]}`; `GET /contexts/{id}` → context or 404; `POST /ask|/step|/escalate` accept `user, context_id, session_id`; every response gains `request_id, context_id`.
- Tests: `app/test_governance.py` (stdlib `unittest`, no pack): unknown→None, header fallback, crc slot stable, LRU eviction, 409 on foreign context. Endpoint tests in `test_app.py` `check()` style (needs pack): `handle_ask({"user":"nobody"}) == (400, …)`.

## 2. KV cache

- Flags in both `_ensure`: `-np N -c 4096*N --cache-reuse 256 --metrics` (4B); `-np N -c 12288*N --cache-reuse 256 --metrics` (80B, N default 1 — RAM). Per-request body: `"cache_prompt": true, "id_slot": slot` (llama-server honours both on `/v1/chat/completions`). Never pass `--no-cache-prompt`. `--slot-save-path` deferred (D-log): contexts are memory-only, slots few.
- Read-back per call: `usage.prompt_tokens`, `usage.completion_tokens`, `timings.prompt_n` (tokens actually evaluated), `timings.cache_n` when present. `kv.cached = timings.get("cache_n", prompt_tokens - prompt_n)`, `kv.evaluated = prompt_n`.
- `cache_hit_ratio(tier, model) = Σ kv.cached / Σ (kv.cached + kv.evaluated)` over the window (PromQL from the two counters). FAQ cache stays `Engine.cache` → event field `faq_cache: hit|miss`, counter `codemap_cache_requests_total{cache="faq"}` — separate name, separate counter.
- Test (`unittest`): `Meter.add(fake_response)` with/without `cache_n`; mocked loop shows `meter.calls == steps`.

## 3. Token + credit accounting

- `telemetry.Meter`: `.add(usage: dict, timings: dict | None)` accumulates `prompt, completion, cached, cache_creation, evaluated, calls`; `.snapshot()`. Fed by `rung_cpu.chat`, `big_tier.chat`, `api_tier._call` callers (Anthropic: `input_tokens→prompt`, `output_tokens→completion`, `cache_read_input_tokens→cached`, `cache_creation_input_tokens→cache_creation`).
- `R/app/credits.json` `{schema:1, unit:"credit", per_1k:{"local-4b":{in,out}, "local-big":{in,out}, "api:<model-id>":{in,out}}, cache_read_multiplier:0.1, cache_creation_multiplier:1.25, faq_hit:0, engine_step:0}` — relative numbers, no currency (D-log says so).
- `credits.compute(tier, model, tokens) -> float` = `((prompt−cached)·in + cached·in·m_read + cache_creation·in·m_create + completion·out)/1000`, 4-dp round.
- Budget: `metrics.Registry.spent_today(user)` (UTC date from events; rebuilt from `events.jsonl` at boot). Checked BEFORE any model call in `/ask` (non-FAQ) and `/escalate`; `/step` is free. Exhausted → emit `budget_refusal` event, **HTTP 429** `{error:"budget_exhausted", terminal:"budget_exhausted", spent, budget, resets_at}` + `Retry-After` seconds to UTC midnight. Why 429 not 402: no currency, quota semantics, MCP/persona clients already back off on 429. Post-paid: a request that starts under budget finishes; overrun counts.
- `GET /budget?user=` → `{user, date, budget, spent, remaining, requests, resets_at}`.
- Tests (`unittest`): rate-card arithmetic incl. multipliers; registry replay of 3 fixture events → spent; 429 path via `handle_ask` in `test_app.py` with a 0-budget test user injected.

## 4. Feedback

- `POST /feedback {request_id, user, rating?:1..5, vote?:"up"|"down", tags:[…], comment?}` → 201 `{ok, request_id}`. Rules: exactly one of rating/vote; `request_id` must be in the registry's seen-ids ring (rebuilt from events) else 404; tags ⊆ `{wrong, incomplete, hallucinated, slow, great, should_have_passed, should_have_answered}` else 400; comment ≤ 500 chars; user known. Append-only; a repeat (request_id,user) supersedes in metrics.
- UI: `render()` gets `r.request_id`; append `<div class="rate" data-rid>` with five `★` buttons + optional one-line input, `post('/feedback', …)`, swap to "thanks" on 201. ~25 lines vanilla JS, `esc()` reused.
- Test: `test_governance.py` validation matrix; `test_app.py` end-to-end `handle_ask` → `handle_feedback`.

## 5. Metrics surface

- `R/app/metrics.py`: `Registry` (in-memory aggregates, `observe(event)` on every emit, `load(events_path)` at boot — a view of the artifact, restart-safe); `render() -> str` Prometheus text 0.0.4 (HELP/TYPE, label escaping, cumulative `le` buckets). `GET /metrics`, content-type `text/plain; version=0.0.4`.
- Names: `gen_ai_client_token_usage{gen_ai_operation_name="chat", gen_ai_provider_name, gen_ai_request_model, gen_ai_token_type=input|output|cache_read, tier}` histogram (semconv buckets 1,4,16,…,67108864); `gen_ai_client_operation_duration_seconds{…, tier}` histogram (0.01…81.92); `codemap_requests_total{event_type,tier,terminal,user}`, `codemap_credits_total{user,tier,model}`, `codemap_budget_remaining{user}` gauge, `codemap_budget_refusals_total{user}`, `codemap_cache_requests_total{cache=faq|kv, result=hit|miss}`, `codemap_kv_tokens_total{tier,model,kind=cached|evaluated}`, `codemap_feedback_total{tier,rating}`, `codemap_feedback_tags_total{tag}`, `codemap_info{prompt_version,pack_version,model_version,schema}=1`.
- Cardinality: `user` only on counters/gauges, never on histograms; no request_id/context_id/q labels anywhere. Worst case ≈ users×tier×model×terminal ≈ 2k series.
- Tests (`unittest`): golden exposition from `telemetry/fixtures/events.sample.jsonl`; mutation guard — a label named `request_id` fails the test; every `_bucket` series ends in `+Inf`.

## 6. Request identity (`R/app/ids.py`)

`new_request_id()` uuid4; `prompt_version(tier)` = `"v1@"+sha256(master_prompt_v1.txt)[:16]` (4B), `"big@"+sha16(_STORY)`, `"api@"+sha16(navigator_system minus L1 index)`; `pack_version()` = `graph/pack/manifest.json` version else sha16(manifest); `model_version(tier)` = gguf basename + `GGUF_SHA16` (4B, `9c454526d7d0d1b0`), gguf basename (80B), API `model` id returned. All four + `schema` + `slots` in `/status`. Test: sha stable across calls; `/status` carries all keys.

## 7. MCP server for personas (`R/mcp/`, no existing MCP in `plugin/` — only `CLAUDE.md.template`)

- `codemap_mcp.py` stdlib stdio JSON-RPC 2.0, newline-delimited; methods `initialize` (protocolVersion "2025-06-18", capabilities `{tools:{}}`), `notifications/initialized`, `ping`, `tools/list`, `tools/call`. Pure `dispatch(msg, http) -> reply` so tests need no subprocess.
- Tools: `codemap_ask(q, context_id?, tier?)`, `codemap_step(dsl)`, `codemap_feedback(request_id, rating?|vote?, tags, comment?)`, `codemap_status()` → `content:[{type:"text", text:<json>}]`. Every call forwards `user = env CODEMAP_USER` (server refuses to start without it) and `session_id = env CODEMAP_SESSION or uuid4-at-start`; `CODEMAP_URL` default `http://127.0.0.1:7345`.
- `mcp.json`: `{"mcpServers":{"codemap":{"command":"python","args":["applications/CodeMap/mcp/codemap_mcp.py"],"env":{"CODEMAP_USER":"${CODEMAP_USER}"}}}}`.
- Launch: `CODEMAP_USER=persona-newcomer CODEMAP_SESSION=$(uuidgen) claude -p "$(cat personas/newcomer.md)" --model claude-haiku-4-5-20251001 --mcp-config applications/CodeMap/mcp/mcp.json --allowedTools "mcp__codemap__*" --output-format json`. Join key: CodeMap events `user` == `CODEMAP_USER` == Claude Code `OTEL_RESOURCE_ATTRIBUTES=user.persona=<id>` (and the transcript `requestId` dedupe from the facts file); `session_id` joins one run.
- Tests `mcp/test_mcp.py` (`unittest`): `tools/list` names; `tools/call` with fake `http` asserts `body["user"]=="p1"` from env; missing env → exit 2; unknown method → JSON-RPC −32601.

## 8. Event schema (one line, `sort_keys`, `schema:1`)

| field | type | note |
|---|---|---|
| schema | int | 1 |
| event_type | str | ask, step, escalate, feedback, budget_refusal |
| ts | str | ISO-8601 UTC ms |
| request_id | str | uuid4; feedback/budget_refusal reference the target/refused id |
| parent_request_id | str,null | escalate → its /ask |
| session_id, context_id | str,null | |
| user | str | |
| tier | str | faq, local-4b, local-big, api, engine, none |
| model, model_version, prompt_version, pack_version | str,null | |
| protocol | str | cache, model, big, api, descend, step |
| terminal | str,null | answer, pass, stall, invalid, descend, budget_exhausted, error |
| steps, backtracks | int | |
| duration_ms | int | |
| tokens | {prompt, completion, cached, cache_creation: int} | |
| kv | {slot: int,null, cached, evaluated: int, calls: int} | |
| faq_cache | str | hit, miss, n/a |
| credits | float | |
| q, answer | str | answer ≤ 2000 chars; gitignored artifact, judges need text |
| trajectory | [str] | |
| offer_id, reason_category, error | str,null | |
| rating, vote, tags, comment | int,str,[str],str,null | feedback only |
| spent, budget | float | budget_refusal only |

`telemetry.emit(ev)` validates against `SCHEMA` (name→type table), appends under a `threading.Lock` (ThreadingHTTPServer), `open(...,"a")` per call. Path `R/telemetry/events.jsonl`, env `CODEMAP_TELEMETRY_DIR` for tests. Clock-jump law: budgets key on the event's UTC date — a forward jump resets a day; noted, accepted.

## 9. Slices (each = code + tests + one D-line in `docs/07-governance-loop-server-plane.md` + CHANGELOG Unreleased)

| # | Slice | Files |
|---|---|---|
| 1 | Event artifact + request identity + handler extraction | `app/telemetry.py`, `app/ids.py`, `app/server.py` (handle_*, emit, /status), `app/test_governance.py`, `app/test_app.py`, `.gitignore`, `docs/07…md` (D1 artifact, D2 three loops one meter), `CHANGELOG.md` |
| 2 | Users + contexts | `app/users.json`, `app/users.py`, `app/contexts.py`, `app/server.py`, tests ×2, `.gitignore`, D3 (400), D4 (memory-only, no history injection) |
| 3 | Meter + KV flags | `app/rung_cpu.py`, `app/loop_runner.py`, `app/model_client.py`, `app/big_tier.py`, `app/api_tier.py`, `app/test_app.py` (4 mock lines), `test_governance.py`, D5 (`-c`×slots), D6 (slot-save deferred) |
| 4 | Credits + budget + 429 | `app/credits.json`, `app/credits.py`, `app/metrics.py` (Registry only), `app/server.py` (/budget), tests, D7 (no currency), D8 (429) |
| 5 | Feedback + UI widget | `app/server.py`, `app/ui.html`, tests, D9 |
| 6 | `/metrics` exposition | `app/metrics.py` (render), `app/server.py`, `telemetry/fixtures/events.sample.jsonl`, tests, D10 (view of artifact) |
| 7 | MCP + persona launch | `mcp/codemap_mcp.py`, `mcp/mcp.json`, `mcp/test_mcp.py`, `mcp/README.md`, `personas/*.md`, D11 (join on user) |
| 8 | Eval-plane replay gate | `eval/ci/test_telemetry_schema.py` (replays fixture, no model), `eval/ci/README.md` count line, `Readme.md` goal row, D12 |
