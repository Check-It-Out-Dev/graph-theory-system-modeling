# eval/put — prompt under test

Can a team write a prompt for its coding agents and **measure** that the agents follow the rules written in it?
This directory is the pipeline that answers it: a prompt, tasks with hidden acceptance tests, deterministic checks
per rule, a blind judge, GEPA to iterate the prompt, a stop rule for saturation, a held-out certification and a
promotion gate. The long form is `../../docs/09-prompt-under-test.md`; the statistics are `METRICS.md`.

## The pieces

| file | role |
|---|---|
| `instances/<name>/prompt/v1.md` | the seed prompt: XML, each rule a `<rule id>`; `references/` are generated includes |
| `instances/<name>/contract.json` | rule → check → predicate, weights, statistics settings, runner limits — declared before any run |
| `instances/<name>/tasks/tasks.jsonl` | the tasks: text, interface, split (train/holdout), tags, exemplar patterns, hidden tests, pass-to-pass classes |
| `instances/<name>/tasks/<id>/hidden/`, `reference.patch` | what the coder never sees; `validation.json` proves each hidden test fails on the base and passes on the reference |
| `put_runner.py` | one coding run: fresh worktree, restricted Sonnet session with the prompt as CLAUDE.md, budget, diff, tests |
| `put_checks.py`, `put_diff.py` | the deterministic checks, diff-scoped, from committed artifacts only |
| `put_judge.py`, `judge/rubric-r5.md` | the blind Opus judge (r5: sees the unchanged text of the files a diff modifies) |
| `put_calibrate.py`, `judge/` | the judge against reference grades: anchors (`select`, degraded variants via `extend`), `score --rubric rN [--label L]`, `reference-scores.json` (who graded what, every gap argued), `bench/` (the page to grade on) |
| `put_score.py`, `put_stats.py` | the score and the statistics (Wilson, delta, bootstrap, sign test, kappa, AC1, Spearman) |
| `put_gepa.py`, `put_stop.py` | GEPA over the prompt, its memorisation guard and the plateau stopper |
| `put_certify.py`, `put_promote.py` | the held-out verdict, the ceiling test, the promotion gate and the CLAUDE.md branch |
| `run.py` (`put_cli.py`) | one command per mode |
| `RUNS.md` | every campaign: its Actions run, its label, its artifact digest |

## Running it

On the self-hosted runner (the evidence plane): edit `.github/prompt-eval/campaign.json` on the backend branch
`ci/prompt-eval` and push. The workflow `prompt-eval.yml` runs the campaign on `norbert-box`, writes the report into
the job summary and uploads the run directories. Locally, the same command:

    PYTHONUTF8=1 python eval/put/run.py baseline --campaign campaign.json --base-repo <checkout of the repository at base_sha>

| mode | what it does |
|---|---|
| `smoke` | the repository's CLAUDE.md (or the seed) on a task or two |
| `baseline` | the seed on every training task k = 3 times, judged twice: delta and the per-rule rates |
| `gepa` | GEPA until the plateau (K = 3 without a gain above delta) or 60 metric calls |
| `certify` | the seed k = 3 and a candidate k = 4 on all tasks: the hold-out verdict and the ceiling |
| `rejudge` | re-grade a finished campaign's runs (`"of": <label>`) and the degraded anchors twice with the current rubric; re-measures delta; no coder run |
| `report` | rebuild a report from the run directories |

Then `put_promote.py decide --certify <label> --gepa <label>`; `apply --target-repo <backend> --push` commits the
certified prompt as `CLAUDE.md` on a branch of the backend. The owner merges. A closed gate is a result: the decision is
kept as `runs/promote-<date>.json` and the seed stays the team's prompt.

The arc of 2026-09-23/24, in order: `baseline-2026-09-23` (r4) → calibration → `baseline-2026-09-23-r5` (rejudge) →
`gepa-2026-09-24` → `certify-2026-09-24` → `promote-2026-09-24.json` (closed: the hold-out CI includes 0).

## A second instance

An instance is data plus a check registry: write a prompt with `<rule id>` tags, a `contract.json`, tasks with hidden
tests and references, validate them (`put_tasks.py validate`), and give each rule a check. The PR-reviewer prompt of
the backend's `ai-review.yml` is the natural second instance: its tasks are pull requests with their reports, and its
rules (quote every number byte for byte, the fixed section order, the closing line) are already checked
deterministically by that workflow's step I5 — `instances/pr-reviewer/contract.json` sketches the mapping. It is
described, not run.
