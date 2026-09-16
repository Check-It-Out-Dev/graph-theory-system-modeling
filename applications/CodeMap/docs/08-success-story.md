# CodeMap Remote — the success story, with its numbers

_Written 2026-09-16/17 while the last nights ran. Every figure here has one home in a committed artifact named beside it; the decision log (`07-ai-quality-governance.md`, D-R1…D-R17) says why each choice was made. Status: **the green box is earned by two conditions** (a night whose judge is calibrated on its own rows, and a gain measured in thirty baseline pairs) — the section "The night that flipped the row" is filled in when it happens._

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

## What the numbers say (and do not say)

Two nights before the reindex, on prompt v1 then v2:

| night | who | judged | grounded | correct | rating | credits per correct | misses |
|---|---|---|---|---|---|---|---|
| 2026-09-16 | one persona, calibration pass | 35 | 0.91 | 0.74 | 4.5 | 24.6 | 0 |
| 2026-09-17 | all six personas | 54 | 0.87 | 0.78 | 4.16 | 17.5 | 11 |

What they say: the navigator does not hallucinate pointers (every pointer resolves server-side to a known entity; grounded 0.87–0.96 by the judge), abstains honestly on out-of-corpus questions, and the second prompt halved the credits per correct answer. What they do not say yet: that CodeMap beats grep for an agent with a filesystem. That claim is the gain — the same question answered by the same persona with and without the graph — and the first two nights produced zero pairs because the planner never scheduled the partner conversation. Fixed; the campaign counts pairs across nights (`eval/quality/campaign.py`).

Three things the first full night corrected, recorded in `INCIDENTS.md` rather than smoothed over: the same 54 answers judged twice differed by nine points on grounded; the oracle on rephrased persona questions disagreed with the judge (κ −0.28 on 11 rows, mostly abstentions that named the right files — so the oracle now answers WHERE only); baselines had no partners.

## Where the graph earns its keep, honestly

- **Vocabulary mismatch.** The product owner and the newcomer ask in task nouns, not repository nouns; grep has nothing to grep for. This is where the subsystem index, the L2 prose and the curation notes do the work.
- **Cost per first hop.** 16–25 credits for an answer with verified pointers, mostly cache reads over a 17k-token prompt.
- **Where it was thin.** The reviewer persona asking "what depends on X" rated 3.0 and filed nine misses: mocks, cascades, an ownership check the edges did not carry. Those misses are the saturation backlog the extract job reads next, and the full reindex moved the graph two months forward in one step (drift 17/54 bank rows, an honest cost).

## The night that flipped the row

_(to be written from the artifact of the qualifying night: κ_oracle on ≥ 30 of its own rows, the campaign's thirty pairs, the claims rows that gate the README)_

## How to run it yourself

- Ask: `claude mcp add --transport http codemap https://codemap.checkitout.app/mcp --header "Authorization: Bearer $CODEMAP_TOKEN" --header "X-CodeMap-User: owner"`.
- A night: `pwsh -File applications/CodeMap/tools/night.ps1 -Date <YYYY-MM-DD>` (personas, bank pass, judge, calibration, quality, campaign, commit).
- A reindex: `graph/delta/fullscan.py` → `extract.py --full` → `reindex.py propose|apply|reclue|release`.
- A prompt iteration: `modal run eval/optimize/modal_gepa.py --train 6 --val 6 --max-metric-calls 30`, then `promote.py --confirm 24`.
- The numbers: https://check-it-out-dev.github.io/graph-theory-system-modeling/quality/ and the six dashboards linked there.
