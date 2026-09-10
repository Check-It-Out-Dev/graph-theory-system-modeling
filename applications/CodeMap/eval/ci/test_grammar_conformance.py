"""4,936 lines a real model wrote, re-judged by today's grammar.

The frozen runs recorded every DSL line the models emitted, including the ones that failed.
That makes them a conformance suite no hand-written fixture could match: the malformed inputs
are real malformed inputs, produced under a real prompt by a model trying to comply.

The gate is not "everything parses" --- 148 of these lines do not, and should not. It is that
the verdicts are unchanged: the same lines accepted, the same lines rejected, for the same
reason. Loosening the parser to swallow a trailing semicolon would be defensible engineering
and a silent re-write of history, because the ``invalid`` terminals in nine committed
artifacts were decided by the parser as it stood. This test makes that trade visible: change
the grammar and the fingerprint moves, and the artifacts have to be re-scored or retired.
"""

import json

import harness

FROZEN = json.load(open(harness.CONFORMANCE, encoding="utf-8"))


def test_the_corpus_of_recorded_steps_is_intact():
    got = harness.conformance()
    assert got["steps"] == FROZEN["steps"], (
        f"the recorded trajectories changed size: {FROZEN['steps']} → {got['steps']}")
    assert set(got["per_run"]) == set(FROZEN["per_run"]), (
        "the set of frozen runs changed; re-freeze with `python harness.py --freeze` "
        "and say in the commit which run was added or removed")


def test_todays_grammar_reaches_the_frozen_verdicts():
    got = harness.conformance()
    moved = {run: {"frozen": FROZEN["per_run"][run], "today": counts}
             for run, counts in got["per_run"].items()
             if run in FROZEN["per_run"] and counts != FROZEN["per_run"][run]}
    assert not moved, (
        "app/dsl.py now judges recorded model output differently:\n"
        + json.dumps(moved, indent=1)
        + "\nThe nine committed benchmark summaries were scored by the old grammar. Re-score "
          "them, or retire them, before re-freezing.")
    assert got["fingerprint"] == FROZEN["fingerprint"], (
        "the per-step verdicts moved without moving any count --- a step changed class or a "
        "trajectory was reordered:\n"
        + "\n".join(f"  {r} {i} {c}: {s}" for r, i, c, s in harness.unparseable_examples()))


def test_the_rejected_steps_are_still_rejected_for_stated_reasons():
    """Name the failure classes rather than only counting them.

    ``unknown-verb`` is the big tier's ``cypher()`` reaching a parser that does not know it
    --- expected, and handled before the parser in ``app/big_tier.py``. ``not-an-expression``
    is prose leakage and truncated answers. ``wrong-arity`` is an unquoted name with a comma
    in it. A new class appearing here is a new way for a model to fail, and deserves a look
    rather than a re-freeze.
    """
    classes = {c for counts in FROZEN["per_run"].values() for c in counts}
    assert classes <= {"ok", "not-an-expression", "unknown-verb", "wrong-arity"}, classes
    rejected = sum(n for counts in FROZEN["per_run"].values()
                   for c, n in counts.items() if c != "ok")
    assert rejected > 0, (
        "no recorded step is rejected any more --- the suite has stopped testing the grammar")


def test_parsing_a_long_whitespace_run_stays_cheap():
    """The parser sees whatever the model emits, so its worst case is the model's to choose.

    ``^\s*([a-z]+)\s*\(\s*(.*?)\s*\)\s*$`` under ``re.S`` gave the engine two ways to
    account for every space around the arguments --- ``\s*`` or the dot --- and it backtracked
    through the combinations. Measured before the fix: 500 spaces took 28 ms, 1,000 took 218 ms,
    2,000 took 1.7 s and 4,000 took 13.4 s, and 30,000 did not finish inside five minutes
    (py/polynomial-redos). Stripping the whitespace before matching leaves one parse for any
    input; the same measurements are 0.013, 0.020, 0.023 and 0.029 ms.

    The bound below is three orders of magnitude above the fixed cost and three below the old
    one, so it says "not exponential" without being a benchmark that fails on a loaded runner.
    """
    import time

    from dsl import ParseError, parse

    worst = "find(" + " " * 4000
    started = time.perf_counter()
    try:
        parse(worst)
    except ParseError:
        pass
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert elapsed_ms < 100, (
        f"parsing 4,000 spaces took {elapsed_ms:.0f} ms; the pre-fix pattern took 13,405 ms and "
        "grew faster than the input. Something has reintroduced an ambiguous whitespace match.")
