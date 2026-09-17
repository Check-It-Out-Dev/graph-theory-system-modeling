# CodeMap Remote — the success story, with its numbers

_Written 2026-09-16/17 while the nights ran. Every figure here has one home in a committed artifact named beside it, and the ones the README leans on are re-read by `eval/ci/test_published_numbers.py` on every push; the decision log (`07-ai-quality-governance.md`, D-R1…D-R21) says why each choice was made. Status: **green since night 2026-09-20** — a judge calibrated on that night's own rows, and a gain measured on 35 baseline pairs. Green means measured: the pairs say CodeMap, as served today, costs an agent more than grep, and the sections below say why._

## The ask, in the owner's words

> Continuous graph saturation, delta ingestion, and when Grothendieck is doing a partition some kind of interaction on the PR itself to decide. A lightweight application on the small VPS, answering via MCP on my subscription token. Emulate the other Claude Code agents as humans, giving the ratings; evaluate the prompt quality and use automated tools to change it. Each persona with its own credit limits, emulated. Grafana, Loki. Security for a small team, even behind a firewall. Make it SOTA quality, publish the metrics on GitHub Pages, with visual proofs.

## What was built in one day (2026-09-16)

| hour (CEST) | slice | the proof |
|---|---|---|
| 13:20 | goal set; Neo4j is the memory; a feature branch | `~/.claude/plans/GOAL-codemap-remote.md`, Neo4j `CodeMapGovernance2026` |
| 13:40 | the served MCP: JSON-RPC over HTTP, bearer + user enum, seven tools, pointers never file bodies | `remote/`, 401 without a token, 400 for an unknown user |
| 14:00 | the navigator on the subscription: Claude Sonnet through the CLI, the engine as a loopback MCP, sessions resume | a second turn read 28,968 tokens from cache instead of writing them |
| 14:10 | credits without currency, `/metrics`, Influx + Loki push to Grafana Cloud | the rate card, one event line per request |
| 14:20 | the pack as a GitHub Release; CI fetches it and runs the engine check | `pack-1.0.0`, 347 KB |
| 14:35 | hosted at `codemap.checkitout.app` (Docker, nginx, Let's Encrypt), reload without restart | `/healthz` 200 |
| 14:45 | six personas with credit caps, rating only after verifying a pointer in their own checkout | the first conversation: 13 turns, ratings 5/5/4/4, 13.5 credits |
| 14:50 | six public dashboards as code | `observability/grafana/public-urls.md` |
| 14:55 | a judge calibrated against the execution oracle | κ 0.84 on 29 exact questions, 20 frozen anchors |
| 15:05 | one quality artifact per night, replayed in CI | `eval/quality/runs/2026-09-16.json` |
| 15:10 | delta extraction with no model: eligibility, fingerprints, structural edges, churn threshold | the backend's last 15 commits: 6 added, 60 modified, churn 4.7 % |
| **15:26** | **the loop closes**: Grothendieck proposes on issue #4, `/codemap accept`, Release `pack-1.0.1`, PR #5, the VPS reloads in ninety seconds | issue #4 CLOSED, PR #5 MERGED |
| 15:50 | drift measured without a model; reclue of touched subsystems; saturation as a number | `graph/ledger/1.0.1.drift.json` (6/55) |
| 16:30 | GEPA on Modal: the prompt improves on the path it is used on; prompt v2 promoted after a confirmation split | PR #6; the VPS serves `nav@dbafd392…` |
| 16:45 | five governance cards, a NIST map, the Art. 50 line, a CI gate for them; the quality page on Pages | `/quality/` 200 |
| 17:26 | the first full night: six personas, 18 conversations, 58 asks, 11 misses | `eval/humans/runs/2026-09-17.*` |
| 17:45 | PR #3 squash-merged to main; CI, Pages, Sonar green | `main 5fc6c86` |
| 19:05 | **the full reindex on the box**: both repositories at their public heads, 174 entities placed by Grothendieck in chunks, 22 subsystems reclued, Release `pack-1.1.0` served | 1,594 entities, 34 navigators, coverage 100 % of eligible files |
| 20:30 | the judge calibrated on a night's own rows — and κ's paradox under a skewed oracle named, measured, and reported beside AC1 | `eval/judge/runs/2026-09-18.json` (agreement 0.85, AC1 0.82, κ 0.25), D-R18 |
| 20:45 | the pair count corrected (skipped partners are no pair), paired conversations first, one night per UTC day; the cost measured by hand | D-R19, the tables below |
| 21:00 | campaign nights plan past the budget and skip a baseline whose partner the day cannot pay for; the night's events fetched as the server sent them | D-R20 |
| **02:02–05:19 (09-17)** | **the night that flipped the row**: 71 persona conversations, 154 navigator answers judged, 31 pairs in one night | `eval/*/runs/2026-09-20.*`, `eval/quality/runs/campaign.json` |

## What the numbers say (and do not say)

Two nights before the reindex, on prompt v1 then v2:

| night | who | judged | grounded | correct | rating | credits per correct | misses |
|---|---|---|---|---|---|---|---|
| 2026-09-16 | one persona, calibration pass | 35 | 0.91 | 0.74 | 4.5 | 24.6 | 0 |
| 2026-09-17 | all six personas | 54 | 0.87 | 0.78 | 4.16 | 17.5 | 11 |
| 2026-09-18 | six personas + a 30-row bank pass, pack 1.1.0 | 74 | 0.78 | 0.76 | 3.98 | 22.4 | 16 |
| 2026-09-19 | two Sonnet personas + 23 bank rows (a budget-collision night) | 42 | 0.86 | 0.76 | 3.74 | 14.1 | 1 |

What they say: the navigator does not hallucinate pointers (every pointer resolves server-side to a known entity; grounded 0.87–0.96 by the judge), abstains honestly on out-of-corpus questions, and the second prompt halved the credits per correct answer. What they do not say yet: that CodeMap beats grep for an agent with a filesystem. That claim is the gain — the same question answered by the same persona with and without the graph — and the first two nights produced zero pairs because the planner never scheduled the partner conversation. Fixed; the campaign counts pairs across nights (`eval/quality/campaign.py`).

Three things the first full night corrected, recorded in `INCIDENTS.md` rather than smoothed over: the same 54 answers judged twice differed by nine points on grounded; the oracle on rephrased persona questions disagreed with the judge (κ −0.28 on 11 rows, mostly abstentions that named the right files — so the oracle now answers WHERE only); baselines had no partners.

The first condition of the green box was met on 2026-09-18, and not the way the row was written. The judge and the execution oracle agreed on 29 of 34 rows (85 %), the five disagreements all being the judge stricter than the oracle — yet Cohen's κ was 0.25, because the oracle said "yes" on 33 of 34 rows and κ's chance term is computed from those marginals. That is the paradox Feinstein and Cicchetti described in 1990, and there were three ways out: regold the bank (would not help — the oracle already hits), relax the judge's anchors (tuning the instrument to the verdict), or report what κ hides. `calibrate.py` now prints the raw agreement, the prevalence and Gwet's AC1 beside κ, and the verdict has two fixed routes the artifact names: κ ≥ 0.6, or, only when the prevalence is past 85 %, agreement ≥ 80 % with AC1 ≥ 0.6. The thresholds did not move for any night: 2026-09-16 passes on κ (0.84), 2026-09-17 fails on both routes (73 % agreement), 2026-09-18 and 2026-09-19 pass on AC1 (0.82 on 34 rows, 0.78 on 18). The README condition says so in words (D-R18).

## Where the cost goes — the pairs, and a measurement by hand

The pair campaign is the claim that matters to a team: the same persona, the same question, the same checkout, once with grep only and once with CodeMap. The first four real pairs (night 2026-09-18; night 2026-09-19 lost every Haiku and Opus partner to the daily credit budget, D-R19) already did not flatter the graph, and the thirty-one that followed on the flip night confirmed them (next section):

| pair | tokens without / with CodeMap | turns | rating |
|---|---|---|---|
| haiku-ops OD11 | 210k / 363k | 9 / 7 | 5.0 / 5.0 |
| haiku-ops OD30 | 679k / 922k | 22 / 16 | 4.0 / 4.0 |
| sonnet-bugfixer BE01 | 454k / 848k | 7 / 11 | 5.0 / 5.0 |
| sonnet-bugfixer BE29 | 601k / 1,523k | 14 / 20 | 4.0 / 3.7 |

Two meters run in a CodeMap conversation, and the table shows one. The persona's own session re-reads its whole context every turn (97 % of those tokens are cache reads), and every `codemap_ask` answer adds one to two thousand tokens of prose and pointers that compound turn after turn; the persona then opens the pointed files anyway, because rating an unverified answer above 3 is forbidden. The second meter is the navigator on the VPS: a Claude Sonnet session per question, about 200k cached tokens, 8.6k cache-creation and 1.5k output, 5 engine steps and 20 seconds per answer, 16.7 credits — 15.8 M tokens for night 2026-09-18's 75 answers, billed as credits, never in the persona's column.

Underneath both sits the model-free part, and that is where the context engineering lives. Three bank questions answered by hand, both ways, counting the bytes of tool output that enter the context before the answer file is opened (both paths open it):

| question | graph (engine, no model) | grep and reads |
|---|---|---|
| where is the Instagram OAuth callback handled | one `find`, 1.1 KB, the controller named first | a naive `instagram` grep hits 250+ files (the package is `com.sm.instagram`), 26.9 KB of noise; a refined `callback` grep 2.2 KB |
| what stops two instances running the same cron job | two calls, 1.3 KB, plus the 0.5 KB config | `@Scheduled` 2.5 KB, open one job 1.7 KB, grep the lock 1.6 KB |
| what breaks if `AccountStatus.java` changes | one `impact`, 4.7 KB: 50 dependents with relation type and subsystem | 81 files, 671 hits, 9.5 KB of paths with SQL and test noise, no relation types, every import still to read |

Two to twenty times fewer bytes per hop, and the impact question is the one grep cannot answer without reading dozens of files. The 117-node architecture model in Neo4j returned nothing for any of the three; the graph that helps is the 1,594-entity pack. The saving is spent above the engine: agents reach it only through a model that turns 1–5 KB of structure into prose at 210k tokens a call. The finding the pairs point at — expose `find` and `impact` to agents as direct, model-free MCP tools and keep `codemap_ask` for the vocabulary-mismatch questions — is not part of this arc and is recorded here rather than built.

## Where the graph earns its keep, honestly

- **Vocabulary mismatch.** The product owner and the newcomer ask in task nouns, not repository nouns; grep has nothing to grep for. This is where the subsystem index, the L2 prose and the curation notes do the work.
- **Cost per first hop.** 16–25 credits for an answer with verified pointers, mostly cache reads over a 17k-token prompt.
- **Where it was thin.** The reviewer persona asking "what depends on X" rated 3.0 and filed nine misses: mocks, cascades, an ownership check the edges did not carry. Those misses are the saturation backlog the extract job reads next, and the full reindex moved the graph two months forward in one step (drift 17/54 bank rows, an honest cost).

## The night that flipped the row

Night 2026-09-20 started at 02:02 CEST on 2026-09-17, two minutes after the servers' daily credit budgets reset, and ended at 05:19 without a hand on it: the runner planned ten conversations per persona, all but the first paired, and let each persona's published budget stop it (D-R20). 34 baselines and 37 CodeMap conversations ran; 20 baselines were skipped before they started because their partner could no longer be paid for, and 3 partners met an exhausted budget. The navigator answered 154 questions (32.8 M tokens on the VPS, 2,170 credits), the personas left 131 ratings and 36 misses, the judge read every answer, and the script committed the artifacts itself.

**The first condition — a judge calibrated on the night's own rows.** 24 oracle rows (the bank pass plus the personas' bank hits): judge and oracle agree on 22, the oracle says "yes" on 92 %, κ is 0.45 and Gwet's AC1 is 0.90 — calibrated on the skewed-oracle route that D-R18 fixed before this night ran, and on which 2026-09-17 still fails. The night's rates, from `eval/quality/runs/2026-09-20.json`: grounded 0.94, correct 0.90, located 0.95, mean rating 3.89, 15.6 credits per correct answer, oracle success 0.92 — the best night of the five, on pack 1.1.0 and prompt v2.

**The second condition — thirty pairs.** 31 on this night, 35 across nights (`eval/quality/runs/campaign.json`):

| per pair, mean | value |
|---|---|
| persona tokens, baseline ÷ CodeMap | 0.63 (CodeMap conversations used 1.6× the tokens: 38.7 M against 22.6 M) |
| turns, baseline − CodeMap | −3.5 (CodeMap conversations took 3.5 more turns) |
| seconds, baseline − CodeMap | −88.5 (CodeMap conversations ran 88.5 seconds longer) |
| pairs where CodeMap used fewer tokens / fewer turns | 17 % / 26 % |

| persona | pairs | tokens ratio | turns Δ | where it lands |
|---|---|---|---|---|
| haiku-ops | 7 | 0.75 | +0.9 | fewer turns in 4 of 7 — config and flag questions |
| haiku-pm | 6 | 0.79 | −0.7 | closest to even: the vocabulary-mismatch persona |
| sonnet-newcomer | 9 | 0.62 | −4.4 | |
| sonnet-bugfixer | 9 | 0.36 | −9.6 | the most expensive: verifies every pointer, then keeps asking |
| opus-reviewer | 3 | 0.90 | +1.7 | fewer tokens in 2 of 3 — impact questions |
| opus-architect | 1 | 0.56 | −5.0 | |

The two conditions are about measurement, and both are met: the row is green because the quality of a served AI system is now measured by an instrument that is itself calibrated, on enough pairs to say something. What it says is not the story this document was drafted to tell. Served through `codemap_ask`, the graph costs an agent that already has the checkout more tokens, more turns and more time — the persona's context carries every prose answer forward, and the persona still opens the files to verify them. Where it comes close to even is where the draft said it would: the product owner asking in task nouns, the operator asking for a flag, the reviewer asking what depends on a file. And the hand measurement above shows the model-free engine returning two to twenty times fewer bytes per hop than grep. The cost is in the layer between them.

Three things the numbers do not settle, said here rather than left for a reader to find. The two rating columns are not the same instrument — a baseline's rating is the persona's confidence in its own answer, a CodeMap rating is the persona's verdict on someone else's (4.46 against 4.05) — so ratings are reported, not compared. The judge scores the navigator's answers, not the baselines', so correctness is not yet a pair metric. And the personas' own estimate of minutes saved (10.2 on average) is contradicted by the clock (88.5 seconds lost): self-reported savings are not a gain.

What the pairs point at next, and what this arc does not build: expose the engine's `find`, `impact` and `flow` to agents as model-free MCP tools, so the 1–5 KB of structure reaches the agent without a 210 k-token navigator in between, keep `codemap_ask` for the questions a model must translate, and run the same campaign again — the instrument is ready for it.

## How to run it yourself

- Ask: `claude mcp add --transport http codemap https://codemap.checkitout.app/mcp --header "Authorization: Bearer $CODEMAP_TOKEN" --header "X-CodeMap-User: owner"`.
- A night: `pwsh -File applications/CodeMap/tools/night.ps1 -Date <YYYY-MM-DD>` (personas, bank pass, judge, calibration, quality, campaign, commit).
- A reindex: `graph/delta/fullscan.py` → `extract.py --full` → `reindex.py propose|apply|reclue|release`.
- A prompt iteration: `modal run eval/optimize/modal_gepa.py --train 6 --val 6 --max-metric-calls 30`, then `promote.py --confirm 24`.
- The numbers: https://check-it-out-dev.github.io/graph-theory-system-modeling/quality/ and the six dashboards linked there.
