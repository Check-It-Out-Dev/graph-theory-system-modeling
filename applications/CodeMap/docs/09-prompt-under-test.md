# 09 — Prompt under test: can we write a conventions prompt and measure that the agent follows it?

_Arc 5, set by the owner 2026-09-23. Design of record: `~/.claude/plans/GOAL-prompt-under-test.md` (not in this repository). Decision log: `docs/07-ai-quality-governance.md` D-R29 onwards. Code: `eval/put/`._

The question is one: a team writes a prompt that tells coding agents how the code base is built (its conventions and design patterns), hands it to every agent, and wants proof that the agents follow it. This document is the long form of that proof: the pipeline, its statistics, the runs and what they showed. The graph appears once, as one of the rules ("look in the code graph before you edit"); whether the graph pays for itself is not asked here.

## The problem

Every coding agent writes code in its own dialect unless something tells it otherwise. A team that lets several
agents (and several people with agents) into one repository gets field injection next to constructor injection,
schema changes in two styles, side effects inside transactions in one feature and after commit in the next: the
spaghetti the hot 2026 argument is about. The usual answer is a prompt: a `CLAUDE.md` or `AGENTS.md` that states the
conventions. The usual gap is that nobody measures whether the agents follow it. This arc treats that prompt as an
artifact under test: written once, measured per rule, improved automatically, certified on tasks it never saw, and
shipped only through a gate.

## The pipeline, element by element

| element | role | where |
|---|---|---|
| prompt under test | the conventions manual; each rule a `<rule id>`; delivered as the agent's CLAUDE.md | `eval/put/instances/backend-conventions/prompt/` |
| contract | every rule with its deterministic check or judge criterion, the weights, the thresholds — declared before any run | `contract.json` |
| tasks | ten small, real changes in the backend (six training, four held out), each with hidden acceptance tests and a reference solution proven fail-on-base / pass-on-reference | `tasks/` |
| runner | one coding session per (prompt, task, replicate): Claude Sonnet, fresh worktree, restricted tools, output-token budget | `put_runner.py` |
| checks | one deterministic check per rule, scoped to the agent's diff | `put_checks.py` |
| judge | Claude Opus, blind to the prompt, grades correctness, convention fit, design fit, test quality, graph use | `put_judge.py` |
| score | the contract's weighted sum; deterministic share 0.65 | `put_score.py` |
| optimiser | GEPA: reflective prompt evolution with a Pareto pool per task, masked feedback, a memorisation guard | `put_gepa.py` |
| stop rule | K = 3 iterations without a gain above the noise floor delta | `put_stop.py` |
| certification | seed and candidate replicated on all tasks; the hold-out verdict; the ceiling test | `put_certify.py` |
| promotion | a gate, then `CLAUDE.md` on a branch of the backend | `put_promote.py` |
| evidence | each campaign an Actions run on a self-hosted runner, with its job summary and artifacts | `checkitout-backend` `ci/prompt-eval` |

## GEPA in plain terms

GEPA keeps a pool of candidate prompts, starting from the seed. Each iteration it samples a minibatch of training
tasks, runs the current candidate, and collects feedback text: which rules failed, which hidden tests failed, the
judge's reasons with code names masked. A reflection model (Claude Opus through `claude -p`) reads only this
behaviour-level feedback and proposes an edited manual; a guard refuses a proposal that names code from the tasks'
hidden tests, references or interfaces, drops a rule id or an include, or exceeds the size limit. The proposal runs
on the same minibatch and enters the pool only if it beats its parent there; accepted candidates are then scored on
the whole training set. Selection is Pareto per task: a candidate survives if it is the best on at least one task,
so a specialist that fixes one hard task stays alive instead of the pool collapsing to one average winner, and later
proposals can combine what specialists learned. The budget is counted in metric calls (one coding run, its checks and
its verdict = one call); the run stops on the budget or on the plateau rule. What this pipeline adds to stock GEPA:
deterministic per-rule checks feeding the reflection, a blind judge, a reflection prompt that asks for behaviour
instead of facts, the memorisation guard, delta as the acceptance floor of the stop rule, and the ceiling test that
turns "the prompt cannot teach this" into a recommendation for tooling.

## Two prompts, one method

Two prompts in this estate deserve the same treatment: the coding agent's conventions manual, and the CI reviewer
that comments on every pull request (`ai-review.yml`). Both should be tested, and the judge first, because a judge
that is not calibrated turns every later number into its own opinion. This arc runs the coding agent. The reviewer
is the same pipeline with different rules and tasks (`eval/put/instances/pr-reviewer/contract.json` sketches it; its
strictest rule, quote every number byte for byte, is already checked by the workflow's step I5) and is not run here.

## Calibrating the judge: checking its scores against our own

A third of the score comes from an LLM judge, so the judge is graded before its numbers are used. The shape of the
process matters more than who does the grading: a person (here the owner for the scores the owner gave after reading the
evidence, and an independent reviewer the owner delegated for the rest) re-grades a sample of the same changes, blind to
the judge, and every disagreement is argued on the code before anything changes. `eval/put/METRICS.md` §7a is the
normative text; the record is `eval/put/judge/reference-scores.json` and `calibration-r4.json` / `calibration-r5.json`.

| step | what happened here |
|---|---|
| Anchors | 8 baseline runs at even steps of the judge's range. They were all good (the seed writes good code): every reference score for convention and design fit was ≥ 4, κ was 0 on convention fit whatever the judge did. So 4 degraded variants of training tasks joined them, one broken rule each: field injection, a missing lock, a service importing the adapter, an in-transaction listener. |
| Blind grading | Convention fit, design fit and test quality, 1–5 on the judge's own rubric, from the task, the diff, the test results and the unchanged code base. The owner's own decision on the first contested cell (a flawless diff without a new test is a 4, not a 3: test quality charges the missing test) became an instruction for the rest. |
| Comparison, r4 | Pooled over 36 cells: exact 0.72, binarised agreement 0.94, AC1 0.90, Spearman 0.87. Design fit was the weak one: exact 0.58, Spearman 0.65, and every repeated gap had the judge above the reference. |
| Adjudication | The judge was blind to duplication of unchanged code (a new FAQ method repeating `getFaqsByCategory` was praised as reuse), rated convention breaks that are structural defects as mild smells (an unlocked job that runs on every instance), docked a package the task itself mandated, and charged a missing test twice. Differences of taste (4 vs 5 on test quality) were left alone. |
| Revision, r5 | The judge now receives the unchanged text of the files a diff modifies; the design anchors name duplication against it and structural breaks; a missing test costs at most one point of convention fit; mandated placement is not graded. Spot-checked on the three anchors it should move and one it must not, then the baseline's 18 runs and the 4 degraded anchors were re-judged twice on the box. |
| Comparison, r5 | Design fit exact 0.92, Spearman 0.97, mean difference 0.08; pooled exact 0.83, Spearman 0.92; the judge's own test-retest noise halved (MAD of score 0.017 → 0.009). Binarised AC1 fell 0.90 → 0.85: of the three cells that now cross the ≥ 4 line, two flip back on the judge's second pass, and on one (a service importing an adapter) the judge counts the broken rule and still gives 4 — which the deterministic `ports_adapters` check fails regardless. δ re-measured: 0.030. |
| Stop | r5 is frozen. A third revision fitted to the same 12 anchors would make them training data for the rubric; the next revision needs new anchors. |

What this calibration cannot show: the reviewer and the judge are one model family, which inflates agreement where
they share a blind spot; 12 anchors will not surface an error on a rare kind of change (a controller holding business
logic, a parallel mechanism beside an existing one); and the owner graded the cells the owner chose, not all 36. In a team
the reviewer is the engineer who owns the conventions, the sample is refreshed from each campaign's runs, and the
same five steps run whenever the rubric or the judge model changes.

## Results (2026-09-24)

| campaign | Actions run | what it says |
|---|---|---|
| `baseline-2026-09-23` | 35885709905 | the seed on the 6 training tasks × 3: 0.911 (r4), every hidden test passed, `tests_written` 12/18 the gap, delta 0.030 |
| calibration | — | 12 anchors; r4 soft on design; r5 written; the record in `eval/put/judge/reference-scores.json` |
| `baseline-2026-09-23-r5` | 35986676711 | the same runs re-judged under r5: 0.907, delta 0.030, judge noise halved |
| `gepa-2026-09-24` | 35987646721 | 0.917 → 0.979 in-sample, four candidates, plateau after iteration 3, 54 metric calls |
| `certify-2026-09-24` | 35995327027 | hold-out 0.902 → 0.965, gain 0.063, 95 % CI −0.007 to 0.135: not certified |
| `promote-2026-09-24.json` | — | the gate closed on the interval; the seed stays |

What was learned, beyond the verdict: a seed transcribed carefully from the team's own CONTRIBUTING already reaches
the ceiling on correctness and on most rules, so the room an optimiser has is narrow and sits where the written rules
are vague (here: tests). GEPA filled that gap — `tests_written` 18/30 → 37/40 at certification — and on the tasks the
seed already did well it bought nothing and cost a longer manual and one scope creep. Four held-out tasks cannot
separate a gain of that shape from noise; the next certification needs more of them, declared before it runs. The
rules the prompt could not teach to the 0.90 bound (an example read first, a new unit test with every change, scope)
are the ones to move into a session hook or a CI gate: that recommendation is the practical output for a team.

## S0 — provisioning and probes (2026-09-23)

Every item below was run, not assumed. Raw transcripts: `eval/put/probes/s0/`.

| Item | Result |
|---|---|
| Models behind the aliases | `--model sonnet` → `claude-sonnet-5` (the coder); `--model opus` → `claude-opus-5-5` (judge, reflector). Read from the `system.init` event; pinned for the arc, stored per run. |
| Bash under `--restricted` | Present when `--tools` names it. **Not confined**: with Bash allowed wholesale, `echo probe > ../x` wrote outside the worktree. The Write tool is confined ("outside … `--restricted` confines the file tools to the working directory"). |
| A narrow Bash allowlist under `--permission-mode dontAsk` | `--allowedTools "Bash(./mvnw *),Bash(./mvnw.cmd *),Bash(git status*),Bash(git diff*)"` admitted `./mvnw -v`, `./mvnw.cmd -v`, `git status`; **denied** `echo … > ../x`, `rm`, `python -c`; read-only `ls` passed (Claude Code auto-allows read-only commands). So the coder runs with that allowlist, never with bare `Bash`. |
| JAVA_HOME in the tool's shell | visible (`C:\Users\Norbert\.jdks\corretto-21.0.7`, set for the user and the machine) |
| Worktree lifecycle (Windows) | `git worktree add --detach` 0.5 s; `remove --force` 0.2 s clean, 0.7 s with `target/` |
| Build and test in a fresh worktree | `./mvnw -q -o test-compile` 30 s cold (the repository's Maven build-cache extension restores unchanged modules); one unit test class with `-Ptest -Dtest=…` 23 s. The planned warm-`target/` copy is unnecessary. |
| Evidence plane | Runner group `put-box` (id 3): visibility selected (checkitout-backend only), public repositories allowed, **restricted to one workflow**: `checkitout-backend/.github/workflows/prompt-eval.yml@refs/heads/ci/prompt-eval`. Runner `norbert-box` (Windows, labels `self-hosted,X64,Windows,put`) online; started at logon from the user's Run key (`conhost.exe --headless C:\actions-runner\run.cmd`; a scheduled task needs administrator rights). |
| Campaign trigger | The owner keeps the work on feature branches (2026-09-23 16:28), so the workflow lives on the backend branch `ci/prompt-eval` and a campaign is a commit that edits `.github/prompt-eval/campaign.json` (push trigger with a path filter; no pull-request trigger). No other backend workflow runs on a push to that branch (checked: pushes trigger only on `main`, `greenfield`, `prod`). First run with `mode: none`: plan job green, both campaign jobs skipped. |
| Environment | `ai-review` holds `CLAUDE_CODE_OAUTH_TOKEN` and has no branch policy, so a job on `ci/prompt-eval` can use it. The executor never reads the secret. |
| Disk | 152 GB free on C:; a worktree is 26 MB checked out, ~55 MB with `target/`, removed after its artifacts are captured |
