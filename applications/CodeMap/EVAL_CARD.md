# Evaluation card — how CodeMap Remote is measured

Every rate on a dashboard or in a README traces to a committed artifact and a script that
recomputes it in CI without a model. Last revised 2026-09-24.

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
a dispute (judge and oracle disagree on *located*) goes to the owner's queue. Beside κ every
calibration prints the raw agreement, the prevalence (mean yes-rate of judge and oracle) and Gwet's
AC1, because κ collapses when almost every oracle row is a hit (Feinstein & Cicchetti 1990: 29 of
34 identical verdicts scored κ 0.25 on 2026-09-18 under a 90 % yes-rate). The verdict has two fixed
routes, and the artifact names the one that held (`basis`): κ ≥ 0.6; or, only when the prevalence
is outside 15–85 %, agreement ≥ 80 % and AC1 ≥ 0.6. Neither threshold moves per night, and a night
with 73 % agreement (2026-09-17) stays uncalibrated on both routes.

## Metrics

Names are the dashboards' names (`observability/grafana/`), one home each: `codemap_grounded_rate`,
`codemap_correct_rate`, `codemap_abstention_rate{honest|false}`, `codemap_pointer_verified_rate`,
`codemap_rating_mean`, `codemap_credits_per_correct_answer`, `codemap_judge_kappa`,
`codemap_gain_*` (vs baselines sharing the seed), `codemap_version_drift_rate`,
`codemap_graph_coverage_ratio{repo}`, `codemap_miss_total{repo}`. Nightly values are pushed as
`codemap_quality_*` gauges; server counters stay `codemap_*`.

## Not gated

A night's judge κ is reported per night beside the calibration pass on exact questions (κ 0.84, n 29): on the first full night it was −0.28 on 11 rephrased rows, mostly abstentions that named the right files, so the night is recorded as uncalibrated and no README row rests on it; night 2026-09-18 (pack 1.1.0, 34 oracle rows from a bank pass and the personas) is calibrated on the AC1 route (agreement 0.85, AC1 0.82, κ 0.25 under a 0.90 yes-rate; the five disagreements are all the judge being stricter than the WHERE oracle). Judge repeatability is not yet measured: the same 54 answers judged twice differed by nine points on grounded (INCIDENTS.md). Gains are gated from night 2026-09-20: a pair is a baseline and a CodeMap conversation of the same persona on the same seed, both run with usage, and 35 exist across nights (`eval/quality/runs/campaign.json`); a baseline's rating is the persona's confidence in its own answer and is not compared with a CodeMap rating, and the judge does not score baselines, so correctness is not yet a pair metric. Night 2026-09-20 is calibrated on the AC1 route on 24 of its own rows (agreement 0.92, AC1 0.90, κ 0.45). The single-night rates above are
descriptive, not a claim of trend; the Qwen signal is reported, not gating; the optimiser's
validation scores are on ≤ 8 examples per run and decide a promotion, not a README row. README
rows flip from 🟡 to ✅ only with an artifact and a check that re-reads it (`eval/ci/`).

## Prompt under test — a conventions prompt for a coding agent (2026-09-24)

The same discipline applied to a prompt that writes code rather than one that answers questions
(`eval/put/`, method in `docs/09-prompt-under-test.md`, normative statistics in `eval/put/METRICS.md`).

| | |
|---|---|
| Corpus | 10 tasks at `ff43730b` of checkitout-backend, 6 train / 4 hold-out, split fixed before the first run (`eval/put/instances/backend-conventions/tasks/tasks.jsonl`); hidden acceptance tests per task, validated fail-on-base / pass-on-reference twice (40/40 arms) |
| Oracles | hidden tests (correctness); 17 deterministic checks scoped to the agent's diff and trace, each proven on a fixture that breaks it; pass-to-pass classes |
| Judge | Claude Opus, rubric r5, blind to the prompt; sees the unchanged text of the modified files. Calibrated on 12 anchors (8 baseline runs, 4 degraded variants) against blind reference grades: pooled exact 0.83, AC1 0.85, Spearman 0.92; design fit exact 0.92 (r4: 0.58). Test-retest MAD of score 0.009 |
| Noise floor | δ = 0.030 (agent replicate variance dominates) |
| Verdict rule | hold-out gain > δ, paired-bootstrap 95 % CI excluding 0, no obligatory rule lost — declared before the first run |
| Result | GEPA 0.917 → 0.979 in-sample; certification hold-out 0.902 → 0.965, gain 0.063, CI −0.007 to 0.135: **not certified**, promotion gate closed |
| Evidence | Actions runs on the self-hosted runner, artifacts committed under `eval/put/runs/`, listed in `eval/put/RUNS.md`; README figures in the `put` section of `eval/ci/claims.json` |

Not gated: the code-base census that motivates the diff scope (7 of 40 `@Version`, 9 of 23 locks,
69 `@Autowired`) is a property of the repository under test, not of an evaluation artifact. The
calibration reviewer is the same model family as the judge; the owner graded the first contested
cell and delegated the rest, and `eval/put/judge/reference-scores.json` records who graded each.
