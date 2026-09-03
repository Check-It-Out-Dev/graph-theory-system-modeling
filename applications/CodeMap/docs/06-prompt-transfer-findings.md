# 06 — Can a foreign model walk the graph? The prompt-transfer ladder, measured

One question from the CodeMap arc deserved its own numbers: our 4B navigator is *trained* on
this graph. What does a large open-weights instruct model do on the same 104-question ladder with
**no training at all**, taught only by prompt? (Qwen3-30B-A3B-2507 and Qwen3-Next-80B-A3B, Q4
quantisations, plain CPU, grammar off unless noted; the same engine, the same fingerprint referee,
the same abstention rules.)

| Prompt rung (cumulative) | 30B route success | What moved |
|---|---|---|
| bare guard (verbs + contract) | 0.031 | 73/104 stalls, repeat loops |
| + GBNF grammar | 0.092 | invalid outputs 8 → 1: format errors disappear |
| + **Cypher-analogy anchor** | 0.169 | **5.5× over bare**; backtracks halved |
| + **step-budget awareness** | **0.246** | stalls 73 → **0**; honest passes back up |

The trained 4B holds **0.9752** on the same ladder: canonical routes are learned, not promptable
at this scale. But two transferable lessons fell out:

**1. Prompting is circuit activation, not instruction.** The Cypher anchor — a table mapping each
CMDSL verb to the MATCH shape the model already knows — was the largest single lever (5.5×). We
did not teach the model anything; we *addressed* pretrained graph-query circuits and attached our
surface to them. The same anchor primed both models to terminate statements with `;`, which our
parser rejected: 17 out of 104 *correct* answers died on one character until the harness learned
to strip it. The instrument failed before the model did; the model was right.

**2. Route referees undercount foreign models.** By fingerprint matching, the 80B looked worse
than the 30B. An entity-overlap content referee (computed, not judged by eye) reversed the
verdict: 80B answers carry an F1 p50 of 0.40 with 49% solid groundings, versus 0.25 and 40% for
the 30B. Large models reach true answers by non-canonical routes, like a strong engineer who
ignores local conventions. Pair every route metric with a content metric before concluding
anything.

Corollary for the applications/ layer: the graph pays off regardless of the navigator's size. A
4B trained for $25 owns its home graph; an untrained 80B gets a quarter of the way in on prompt
alone and is the most honest abstainer of the family (pass-on-unanswerable 0.58). Full data:
`training/data/BENCH_*.json`; harness: `app/bench_promptonly.py`.

## Rung 5 (2026-09-03): the graph-native rung — cypher() in the action space

If prompting is circuit activation, the endgame is handing the circuits their OWN language,
executable. `app/big_tier.py` now ships it: the untrained 80B as a first-class LOCAL tier whose
prompt teaches how the graph was BUILT (indexer → 17 typed edge kinds → Leiden → curation →
clues, field by field), carries the L1 index in the system prompt, and adds `cypher(<stmt>)`:
read-only openCypher on the pack's LadybugDB next to the 13 verbs. Measured on the same 104-gold
ladder (IQ4_XS, strict harness, no synthesis rescue) against the rung-4 budget prompt:

| | budget rung | native rung | what it says |
|---|---|---|---|
| route success (65 canonical) | 0.246 | 0.154 | non-canonical routes cannot fingerprint, by construction |
| pass-on-unanswerable | 0.58 | **0.74** | the honest abstainer got MORE honest |
| number F1 p50 (content) | 0.31 | **0.40** | counts sharpen when counting is a query |
| entity F1 mean (content) | 0.31 | 0.27 | parity within the sample shift (49 → 35 answers scored) |
| solid grounding | 0.92 | 0.92 | nothing invented, either way |
| backtrack rate | 0.17 | **0.07** | native expressivity removes thrashing |
| sessions using cypher | — | 0.28 | conditional aggregates, typed-edge filters, top-k: shapes NO verb covers |

(One iteration inside the rung, both runs committed: repeating the one-action format rule at the
END of the prompt, the recency slot, converted narration invalids 12 → 5 and stalls 11 → 8; entity
F1 was diluted as the weaker rescued answers entered the scored sample. Lost-in-the-middle applies
to your OWN system prompt.)

Three lessons this rung adds:

1. **Verb-shaped ladders undercount native tiers.** The 104 golds were authored as canonical verb
   routes, so the ladder structurally favours the surface the 4B was trained on. The native
   tier's wins — count by type, group by, entry-point filters, exact grounded numbers in 4 steps —
   live OUTSIDE the gold distribution. Pair every ladder with off-distribution probes before
   concluding anything about capability.
2. **Referees must count what the prompt shows.** Injecting the L1 index into the system prompt
   made grounded subsystem citations look invented: the grounding referee read 0.52 until its
   evidence base included the injected index (true value 0.94). Once again the instrument failed
   first, this time the referee.
3. **Dialect rules belong in the prompt, bisected on the engine.** The model's flawless-looking
   `count(CASE WHEN ...)` silently returned garbage (a Ladybug aggregate bug, now recorded in
   DIALECT_NOTES); one prompt line converts the trap into two WHERE-filtered queries.

Product consequence, shipped: the 4B keeps the MFQ-shaped questions; the large local tier is the
private, free escalation for aggregation and filter shapes and the family's most honest pass; and
the consent card stays what it always was, a CLOUD boundary, not a model boundary. Bench:
`app/bench_bigtier.py`; data: `training/data/BENCH_qwen80b-native{,2}.json`.
