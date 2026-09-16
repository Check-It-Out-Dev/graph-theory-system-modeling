# Threat model — CodeMap Remote

Scope: the served MCP on the demo VPS, the delta pipeline in GitHub Actions, the optimiser on
Modal. Security is designed to serve a small team, even behind a firewall — that is why all users
can switch names: identity is a header from an enum and one shared token. A full version needs a
separate account for each person (OAuth 2.1 as the MCP specification describes); out of scope here.
Mapped to the OWASP Top 10 for LLM Applications (2025). Last revised 2026-09-16.

## Trust boundaries

1. Client → nginx (TLS, rate zones) → server (bearer token, user enum) → `claude -p` on the subscription → engine MCP (read-only over the pack).
2. Product repositories → GitHub Actions (no secret in extract; the subscription token only in the `graph-delta` environment, same-repository events) → issue → a person's decision → Release.
3. Server → Modal endpoints (embeddings, reranker) over HTTPS — the endpoints hold no request auth (risk R1).

## OWASP LLM Top 10 (2025) → controls

| id | risk | control in place | residual |
|---|---|---|---|
| LLM01 Prompt injection | code or names crafted to steer the navigator | the navigator never reads file bodies; its tools return structural rows; entity ids the pipeline adds pass an id checker and a person's decision before they reach the prompt as notes | a hostile file name can reach a curation note after a person accepts it — notes are data in the prompt |
| LLM02 Sensitive information disclosure | answers leaking secrets or private data | pointers, not content (`remote/pointers.py`); the pack holds names and structure; events hold questions and answers about code | one shared token: anyone with it reads any persona's conversation context |
| LLM03 Supply chain | a tampered action, package or pack | every GitHub action pinned to a SHA; `real_ladybug`, `gepa`, `@anthropic-ai/claude-code` pinned; pack Releases fetched with sha256 verification and staged before swap | Claude Code itself updates on the VPS image rebuild |
| LLM04 Data and model poisoning | poisoned partition or prompt | the pack changes only by a decision (`propose ≠ apply`), recorded bi-temporally; the prompt changes only through the promotion gate and a pull request; both leave a ledger | a `/codemap accept` by any collaborator; the 48 h timeout accepts silently (recorded as such) |
| LLM05 Improper output handling | answers executed or trusted blindly | pointers resolve server-side to known entities only; the client verifies in its own checkout; nothing from an answer is executed anywhere | — |
| LLM06 Excessive agency | the model doing more than navigating | `--allowedTools mcp__engine__*` only; `engine_cypher` read-only with a blocklist and LIMIT caps; at most 10 turns; personas run `dontAsk` with a fixed tool list | — |
| LLM07 System prompt leakage | the prompt exposed | the prompt is public by design (`prompts/navigator/`), versioned and hashed | — |
| LLM08 Vector and embedding weaknesses | index poisoning, open endpoints | the search index is rebuilt from the pack per version; endpoints are called server-side only | **R1** the two Modal endpoints accept unauthenticated requests (owner action: proxy auth) |
| LLM09 Misinformation | confident wrong answers | judge + execution oracle + abstention measured; ratings without pointer verification capped at 3; disputes to a person; drift measured after every pack change | a single-night sample |
| LLM10 Unbounded consumption | cost and window exhaustion | credits per user per day with 429 + Retry-After; 2 concurrent navigators, queue 8, 503 beyond; nginx rate zones; Modal spend cap; subscription rate limits end a night as `partial` | a determined insider with the token switches names (declared) |

## Secrets

The subscription token has three homes — the VPS `.env`, the GitHub environment `graph-delta`,
the Modal secret `claude-oauth` — and no fourth: never a log, a comment, a dashboard, a commit.
`ANTHROPIC_API_KEY` is stripped from every `claude -p` child (`app/claude_cli.py`). CI gates hold
no secret; the only model-calling jobs run in the `graph-delta` environment on same-repository events
with every person-typed value passed through `env:`.

## Open risks (owner actions)

| id | risk | action |
|---|---|---|
| R1 | open Modal endpoints | put them behind proxy auth or Modal's web-endpoint auth |
| R2 | one-year OAuth token | rotate after the arc (the three homes above) |
| R3 | Cloudflare token scope | scope the DNS token down to the two zones' records |
| R4 | the public backend main is 24.9 % ahead of the pack's indexed base | a full reindex on the owner's box, or accept a raised base for delta digestion |
