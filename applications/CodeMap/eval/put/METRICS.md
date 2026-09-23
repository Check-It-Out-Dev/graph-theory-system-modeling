# Prompt under test — the metrics and their statistics

Normative for `eval/put/`. The definitions below were committed before the first coder run of the instance
`backend-conventions` (arc 5, 2026-09-23) and are never moved after a result is seen. The code that implements each
one is named beside it; `remote/tests/test_put_*.py` holds the known values it is tested against.

## 1. What is measured

A **prompt under test** is a manual a team hands to its coding agents (here: the checkitout-backend conventions, as
the agent's `CLAUDE.md`). An **instance** is (prompt, tasks, contract, rubric, weights). A **run** is one coding
session of one task: Claude Sonnet in a fresh worktree of the repository at the base commit, the candidate prompt as
its CLAUDE.md, the code graph as its one MCP tool, a narrow Bash allowlist, and an output-token budget
(`put_runner.py`). Every run leaves its diff, the files as the agent left them, its tool calls, its test results and
its usage.

## 2. Per-rule compliance (`put_checks.py`, `put_score.rule_rates`, `put_stats.wilson`)

Each rule of the prompt (`<rule id>`) has one deterministic check in `contract.json`, scoped to the agent's diff: the
repository follows its own rules only in part (7 of 40 entities with `@Version`, 9 of 23 `@Scheduled` methods with a
lock, 69 fields injected with `@Autowired` at the base), so a check of the whole tree would score the neighbours. A
check returns a value in [0, 1] and says what it counted. A rule that does not apply to a task (tags) leaves that
task's denominators.

A run **passes** a rule only at value 1; partial credit enters the score, never a compliance rate. The rate over the
applicable runs is p̂ = passes / n, with the Wilson 95 % interval

    ( p̂ + z²/2n ± z·√( p̂(1−p̂)/n + z²/4n² ) ) / ( 1 + z²/n ),   z = 1.96.

A rule is **obligatory** when its lower Wilson bound is at least 0.90. With no failure the bound is n/(n+z²), so the
claim needs n ≥ 35 runs; one failure needs about 58. Rule of three: 0 failures in n runs bounds the failure rate at
about 3/n. Only the rules that apply to every task reach n = 40 in a certification (10 tasks × 4 runs); the
task-specific rules (a lock, a changeset, a port) are **reported-only**, with their interval.

## 3. The score (`put_score.py`)

    S(run) = Σ wᵢ · mᵢ,   mᵢ ∈ [0, 1]

| component | kind | mᵢ | w |
|---|---|---|---|
| hidden_pass | deterministic | hidden acceptance tests + pass-to-pass tests passed / total (0 on a red build) | 0.30 |
| conventions_det | deterministic | mean of the applicable convention rules | 0.15 |
| tests_written | deterministic | the agent's own unit tests: Mockito without a Spring context, compile, pass (0 on a red build) | 0.10 |
| process_det | deterministic | mean(graph_first, exemplar_read) | 0.10 |
| convention_fit | judge r4, (s−1)/4 | against the WRITTEN conventions, never the neighbours | 0.10 |
| design_fit | judge r4, (s−1)/4 | cohesion, layering, reuse, no duplication: the anti-spaghetti score | 0.10 |
| correctness | judge r4, (s−1)/4 | the task's stated behaviours | 0.05 |
| test_quality | judge r4, (s−1)/4 | the agent's tests pin the behaviour | 0.05 |
| graph_use | judge r4, (s−1)/4 | the graph shaped the navigation | 0.05 |

Deterministic share 0.65, judge share 0.35. Hidden tests weigh most because a compliant change that does not work is
worth nothing. Compliance is not left to the judge: an LLM cannot count annotations across a diff reliably, it would
reward imitation of non-compliant neighbours, and deterministic rules give per-rule feedback and a per-rule ceiling.
A candidate's score is the mean over tasks of the mean over replicates. A run without a valid verdict scores its
judge part as 0 and is counted in `unjudged`.

## 4. The noise floor δ (`put_cli.noise`, `put_stats.delta`)

- δ_judge: the seed's baseline runs judged twice, the second time in shuffled order; the mean absolute difference of S.
- δ_agent = z · s · √(2 / (T·k)), s the pooled SD of S over the k replicates of the seed on T training tasks: the
  half-width of a 95 % CI of the difference of two set means at that replication.
- **δ = max(δ_judge, δ_agent)**. Every gain is written beside δ: "gain g (δ = …)".

## 5. Saturation (`put_stop.PlateauStopper`, `put_certify.ceiling`)

GEPA stops when **K = 3** consecutive iterations (proposals, accepted or rejected) bring no validation gain above δ
over the best score so far; the anchor moves only on a gain above δ, and it is persisted so a resumed run keeps its
streak. The stop reason is recorded: `plateau`, `budget` (60 metric calls) or `owner`.

The **ceiling** is judged at certification, where the counts are large enough: when every rule that applies to all
tasks is obligatory for the candidate, the run is `saturated_at_ceiling`; otherwise `saturated_below_ceiling`, and
the rules below the bound are published as **the rules the prompt could not teach**, each with the enforcement that
should carry it instead (a session hook, an ArchUnit rule, a CI gate). That split is the practical result: what a
prompt can make an agent do reliably, and what belongs in tooling.

## 6. Generalisation (`put_certify.verdict`, `put_stats.paired_bootstrap`, `put_stats.sign_test`)

Ten tasks, **six for training and four held out**, fixed in `tasks.jsonl` before the first run. Held-out tasks enter
nothing but certification: not GEPA, not the judge's noise, not the reflector. Training scores are best-of-k selected
and in-sample; they are reported and labelled optimistic.

Certification runs the seed 3 times and the candidate 4 times on every task. The hold-out gain dₜ is, per held-out
task, the candidate's mean minus the seed's mean. Its 95 % CI comes from a paired bootstrap: 10,000 resamples (seed
20260923), each drawing the four tasks with replacement and, within each drawn task, each arm's replicates with
replacement, then averaging dₜ; the CI is the 2.5th and 97.5th percentiles. A one-sided exact sign test over the
tasks is reported beside it; with four tasks its smallest p is 0.0625, so it is descriptive.

**The prompt works** when the candidate's hold-out mean exceeds the seed's by more than δ, the bootstrap CI excludes
0, and no rule obligatory for the seed stops being obligatory. Any part failing falsifies the claim, and the report
says which part failed.

## 7. The judge (`put_judge.py`, `judge/rubric-r4.md`)

Claude Opus, one run at a time, blind: it sees the task card, the conventions as the team wrote them (the rule texts
of the seed v1, fixed for the arc), a summary of the agent's work from its tool calls, the measured test results, and
the diff (at most 40,000 characters). It never sees the candidate prompt, its hash or its iteration. It returns five
scores 1–5 with anchors at 5, 3 and 1, a reason of at most 40 words each, and three counts.

Calibration against a human: runs from the baseline, chosen to span the judge's range, are scored by the owner on the
same five criteria. Per criterion, binarised at ≥ 4, the report gives the agreement p_o, the prevalence (yes-rate),
Cohen's κ = (p_o − p_e)/(1 − p_e) with p_e = Σ_c p₁c·p₂c, and Gwet's AC1 = (p_o − p_e^γ)/(1 − p_e^γ) with
p_e^γ = 2π(1 − π), π the mean yes-rate; and Spearman's ρ on the raw scores. κ alone collapses under a skewed
prevalence (a lesson of this repository's navigator judge, D-R18), so the four numbers travel together.
Repeatability is the same batch judged twice in shuffled order (δ_judge above). A change of rubric re-measures both.

## 8. What the numbers do not show

Tasks are synthetic and come from one repository; the coder is Claude Sonnet and the judge Claude Opus on a
subscription, so runs are dated and a model change is a new label; δ is a floor, not a variance model; four held-out
tasks give a wide interval. Whether the code graph beats working without it is not measured here.
