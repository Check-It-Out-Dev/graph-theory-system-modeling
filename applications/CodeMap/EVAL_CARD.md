# Evaluation card — how CodeMap Remote is measured

Every rate on a dashboard or in a README traces to a committed artifact and a script that
recomputes it in CI without a model. Last revised 2026-09-16.

## Corpus

| set | size | home | use |
|---|---|---|---|
| Bank | 104 questions with gold answers and execution fingerprints | `graph/pack/mfq.jsonl` | FAQ tier, oracle, drift, optimiser training/validation |
| Off-distribution probes | 30 (vocabulary mismatches, foreign codebases, ambiguous asks) with an expected terminal | `eval/q/probes_offdist.jsonl` | abstention honesty, optimiser validation |
| Persona conversations | one night = 6 personas × 2–4 conversations × 3–5 turns, ratings after every answer | `eval/humans/runs/<date>.jsonl` | happiness, verified-pointer rate, gains |
| Baselines | a 20 % seeded sample of the night's questions answered without CodeMap | same file, `mode: baseline` | gains (tokens, turns, seconds, correctness) |
| Events | one line per request from the server | `telemetry/events.jsonl` on the VPS; fixtures in `telemetry/fixtures/` | every server-side rate |

## Oracles

- **Execution oracle** (`eval/judge/judge.py: oracle`): for a bank where-question, a gold entity among the first five pointers; for a probe, the expected terminal. Deterministic, model-free, the reference every judge is calibrated against.
- **Pointer verification**: a persona rates only after opening at least one pointer in its checkout; a rating without `verified` cannot exceed 3 (`remote/feedback.py`).
- **Drift** (`graph/delta/drift.py`): the bank replayed by the engine on two packs, order-free canonical hashes, invalidated rows excluded.

## Judge & calibration

Claude Sonnet, batched ten answers per call, five dimensions (located, grounded, correct, abstain,
helpful) on an anchored 1–5 rubric (`eval/judge/rubric.md`); it never sees ratings before scoring
and personas never see its rubric. Calibration: Cohen's κ of `located ≥ 4` against the execution
oracle on the where-archetypes — **κ = 0.84 on 29 rows** (gate ≥ 0.6, `eval/judge/calibrate.py`);
20 frozen anchors detect rubric drift; the Qwen reranker (Modal) is reported as a second family;
a dispute (judge and oracle disagree on *located*) goes to the owner's queue.

## Metrics

Names are the dashboards' names (`observability/grafana/`), one home each: `codemap_grounded_rate`,
`codemap_correct_rate`, `codemap_abstention_rate{honest|false}`, `codemap_pointer_verified_rate`,
`codemap_rating_mean`, `codemap_credits_per_correct_answer`, `codemap_judge_kappa`,
`codemap_gain_*` (vs baselines sharing the seed), `codemap_version_drift_rate`,
`codemap_graph_coverage_ratio{repo}`, `codemap_miss_total{repo}`. Nightly values are pushed as
`codemap_quality_*` gauges; server counters stay `codemap_*`.

## Not gated

Gains are `ungated` until a baseline row shares the seed; the single-night rates above are
descriptive, not a claim of trend; the Qwen signal is reported, not gating; the optimiser's
validation scores are on ≤ 8 examples per run and decide a promotion, not a README row. README
rows flip from 🟡 to ✅ only with an artifact and a check that re-reads it (`eval/ci/`).
