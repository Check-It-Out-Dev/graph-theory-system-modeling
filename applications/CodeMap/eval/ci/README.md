# The evaluation gate — model and prompt evaluation on a runner with no GPU

`python -m pytest applications/CodeMap/eval/ci` · 113 checks · 0.2 s · no model, no GPU, no
network, no credential.

## The problem this solves

A navigator model is evaluated by running it: a GPU, a 2.5 GB quantised checkpoint, half an
hour of wall time per configuration. None of that fits a free CI runner, and pinning a CI
gate to a live model would make the build depend on a GPU's availability, a service's uptime
and a sampler's mood. The usual conclusion is that model evaluation cannot be gated in CI.

That conclusion mistakes the model for the evaluation. An evaluation is a corpus, a scorer, a
contract with the model about what it may emit, and a set of numbers published from the
result. The model is the only part that needs a GPU. The other four are text and arithmetic,
they are the parts that rot silently, and they are exactly what CI is good at holding still.

So the work splits into two planes:

| | offline plane | CI plane |
| --- | --- | --- |
| runs | the model | nothing |
| needs | GPU, checkpoint, ~30 min per configuration | 4 vCPU, 0.2 s |
| when | when a configuration is worth measuring | every push |
| output | `training/data/BENCH_<tag>.json` — every question, the trajectory the model emitted, the verdict the scorer gave | pass or fail, and a comparison table in the run summary |

Nine frozen runs are committed: two models (Qwen3-30B-A3B and Qwen3-Next-80B-A3B) across four
prompt configurations, each on the same 104-question bank. 4,936 recorded DSL steps in total.
Those are the raw material.

## The four gates

**Prompt contract** (`test_prompt_contract.py`). A model's action space is whatever its system
prompt says it is. If the prompt and the parser disagree by one verb or one argument, the
model spends its turns being told "not a valid action" and every downstream metric moves for
a reason no metric names. The training pipeline already states this as law — the master prompt
and the DSL version freeze together, before a single training pair is generated. These tests
are that law, executable: the verb table parsed out of `master_prompt_v{0,1}.txt` must equal
`app/dsl.py`'s, arities included, in both directions; every declared call shape must survive
`parse()`; and the parser must still *refuse* what the contract excludes. The large local tier
runs a second contract, and the test pins the difference: it drops `cache` and adds `cypher`,
one verb, with an implementation behind it.

**Replay scoring** (`test_replay_scoring.py`). Every published figure of every frozen run,
re-derived from that run's own rows by a second implementation of the scorer's arithmetic. Two
implementations that agree are evidence; one implementation checking itself is not. Also the
scorer's invariants, checked against the data it wrote: a gold-fingerprint hit only on an
executable question, success on an unanswerable one only by abstaining, a recorded step count
that equals its trajectory.

**Grammar conformance** (`test_grammar_conformance.py`). The 4,936 recorded steps are a
conformance suite no hand-written fixture could match — the malformed inputs are real
malformed inputs, produced under a real prompt by a model trying to comply. The gate is *not*
"everything parses": 148 of these lines do not, and should not. It is that the verdicts are
unchanged, down to a fingerprint over every per-step judgement. Loosening the parser to
swallow a trailing semicolon would be defensible engineering and a silent rewrite of history,
because the `invalid` terminals in nine committed artifacts were decided by the parser as it
stood. The gate makes that trade visible: change the grammar and the artifacts must be
re-scored or retired, deliberately.

**Corpus integrity and published numbers** (`test_corpus_integrity.py`,
`test_published_numbers.py`, `test_self_description.py`). A benchmark number means "this model, on these questions".
Nothing forces the runs and `eval/q/mfq_all.jsonl` to stay the same corpus, and a question
quietly re-classified would leave every published rate intact and every published rate wrong.
So every run is checked question for question against the bank. Then `claims.json` ties each
figure in the prose to the artifact behind it: the literal text must still be in the document,
and the artifact re-scored today must still round to it at the claim's own precision.

## What is not gated, and why

Two things, both declared rather than left unmentioned.

The **answerable stratum** — which of the 85 executable questions have a canonical reference
derivation — needs the graph pack, which is exported data and not in the repository. Its size
is read from each artifact instead of re-derived. What is available here is that nine
independent runs agree on it, which is the failure that would actually matter: a stratum that
moved between runs would make the ladder's columns incomparable. The unanswerable stratum, by
contrast, *is* re-derived, because the bank alone determines it.

Some **published numbers have no artifact in this repository** — the trained 4B navigator's
own gate table, measured on a test snapshot that lives outside git, and the content-referee
columns computed after the fact over the prompt-only answers. `claims.json` lists them under
`ungated` with the reason. A number without a reason is the thing the manifest exists to
prevent.

## Proof that the gate fires

A gate that has never been red is a claim, not evidence. Seven deliberate breakages, each
restored afterwards:

| mutation | caught by |
| --- | --- |
| the grammar accepts a trailing semicolon | `test_the_grammar_rejects_what_is_not_in_the_contract`, `test_todays_grammar_reaches_the_frozen_verdicts` |
| a verb's arity changes | `test_master_prompt_declares_exactly_the_parser_verbs` |
| a benchmark row is edited | `test_summary_is_what_the_rows_say`, `test_the_artifact_still_supports_it` |
| a published number is edited | `test_the_document_still_says_it` |
| a question is re-classified | `test_the_run_scored_this_bank_question_for_question`, `test_the_unanswerable_stratum_is_the_bank_s_own` |
| this README's own step count goes stale | `test_the_recorded_step_count_is_what_both_documents_say` |
| a sibling suite's markers change, so CI silently runs a different number of tests | `test_the_sibling_mcp_suites_are_counted_as_the_readme_counts_them` |

## Running it

```
python -m pytest applications/CodeMap/eval/ci          # the gate
python applications/CodeMap/eval/ci/harness.py --summary   # the comparison table CI prints
python applications/CodeMap/eval/ci/harness.py --freeze    # re-freeze conformance.json
```

`--freeze` is for when a benchmark run is deliberately added or retired. Re-freezing to make a
grammar change go green is the one move this directory exists to make visible, so say in the
commit message which it was.

## Adding a run

Produce the artifact on the offline plane (`app/bench_promptonly.py` or `app/bench_bigtier.py`
write it), commit it into `training/data/`, re-freeze, and either cite it from a document with
a gated claim or declare it `uncited` with a reason. The replay and conformance tests pick it
up with no further wiring.
