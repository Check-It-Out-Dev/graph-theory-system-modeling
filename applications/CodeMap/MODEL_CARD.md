# Model card — CodeMap Remote (the served navigator)

Aligned with the model-card practice of Mitchell et al. (2019) and the NIST AI RMF Generative AI
profile (2024); "aligned with", never "certified". Every number here has one home in a committed
artifact named beside it. Last revised 2026-09-16.

## Model details

| | |
|---|---|
| System | CodeMap Remote — a code-navigation assistant served as an MCP at `https://codemap.checkitout.app/mcp` (`remote/`) |
| Answering model | Claude Sonnet (`claude-sonnet-5`) invoked through the Claude Code CLI (`claude -p`) on the owner's subscription; `--effort medium`, at most 10 tool turns per answer (`remote/navigator.py`) |
| Optional deep tier | Claude Opus (`claude-opus-5`) for users whose credit budget allows `tier: deep` |
| Tools the model holds | a loopback stdio MCP over the graph engine: `engine_step` (the 13-verb DSL), `engine_cypher` (read-only, LIMIT-capped, blocklisted), `engine_open` (a pointer) — never a file body |
| Prompt | `prompts/navigator/active.md`: template prose + the L1 index + the L2 navigators rendered from the pack + the append-only curation notes; identified by `nav@sha16`; immutable versions `v<N>.md`, log in `prompts/navigator/PROMPT_LOG.md` |
| Second-family signals | Qwen3-Embedding-8B and Qwen3-Reranker-8B on Modal for `codemap_search` and the judge's second opinion; no Claude model scores itself |
| Owner | Norbert Marchewka (Check-It-Out-Dev); the hosted instance runs on the owner's own subscription through the CLI — nobody else's token is ever involved |

## Intended use

Answering "where is it, what depends on it, how does the flow run" questions about the checkItOut
repositories with pointers (path, lines, subsystem, clue) that the asker verifies in their own
checkout. Users are a small team behind one shared token, identified by a name from an enum
(`remote/users.json`); the hosted instance exists to show how prompt quality in an AI system is
measured, observed and improved. A team of ten runs the same server locally with their own
subscription (`remote/README.md`, "Run it locally").

## Out-of-scope

Returning file contents (the server never does); answering about code outside the two indexed
repositories (the navigator abstains — `terminal: abstain`, measured as honest or false abstention);
per-person accounts, billing, or any multi-tenant guarantee; production use beyond a small team
behind a firewall; legal, security or hiring decisions.

## Training data

The answering model is not fine-tuned by this project. What is *built* here is the graph pack the
model navigates (`DATA_CARD.md`): 1,415 entities, 4,786 edges and 92 hyperedges from the July-2026
snapshots of the backend and frontend, curated into 32 subsystems (pack 1.0.0; 1,422 / 33 after
the first delta decision, pack 1.0.1). The offline 4B navigator of CodeMap v1 (`docs/04-training-story.md`)
is a separate artifact and is not deployed here.

## Evaluation

`EVAL_CARD.md` holds the method. Headline artifacts: judge calibration κ = 0.84 against the execution
oracle on 29 where-questions (`eval/judge/runs/2026-09-16.json`); first quality night — 35 answers,
grounded 0.91, correct 0.74, mean rating 4.5, 24.6 credits per correct answer
(`eval/quality/runs/2026-09-16.json`); version drift after the first decision 6/55 bank rows
(`graph/ledger/1.0.1.drift.json`). Public dashboards: `observability/grafana/public-urls.md`.

## Limitations

- The graph is a navigational skeleton; anything the graph does not know is a miss (`codemap_miss`), reported by users and digested later — coverage is measured, not assumed.
- Answers are only as current as the pack Release the server runs; the digestion pipeline lags the product repositories by one decision.
- One shared token: identity is a header, so per-user limits are cooperative, not enforced against a determined insider.
- Subscription rate limits end a night early (`partial: true`) — a measurement of the window, not of the model.
- The judge is a Claude model judging a Claude model; the oracle and the Qwen signal exist because of that, and disputes go to a person.

## Versions

| pack | prompt | since | change |
|---|---|---|---|
| 1.0.0 | `nav@d506112a338ebf22` (v1) | 2026-09-16 | first Release from the workbench |
| 1.0.1 | `nav@e7c2b0c4bb8bbde2` (v1 + 8 notes) | 2026-09-16 | first delta decision (issue #4, `/codemap accept`): 7 entities placed, 1 new subsystem |
