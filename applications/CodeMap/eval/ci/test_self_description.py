"""The gate gates its own description.

Two documents describe this suite in numbers --- the repository README and the one beside
these tests. Numbers in prose are the first thing to go stale, and a page that oversells its
own test coverage is worse than a page with no numbers on it. So the counts are checked
against the data and against the collection itself.
"""

import json
import os

import conftest
import harness

ROOT = os.path.dirname(os.path.dirname(harness.CODEMAP))
TOP_README = os.path.join(ROOT, "Readme.md")
THIS_README = os.path.join(harness.HERE, "README.md")


def _text(path):
    return open(path, encoding="utf-8").read()


def test_the_recorded_step_count_is_what_both_documents_say():
    steps = json.load(open(harness.CONFORMANCE, encoding="utf-8"))["steps"]
    written = f"{steps:,}"
    for path in (TOP_README, THIS_README):
        assert f"{written} recorded DSL steps" in _text(path), (
            f"{os.path.basename(path)} does not say {written} recorded DSL steps; the corpus has {steps}")


def test_the_run_count_is_what_both_documents_say():
    assert len(harness.bench_files()) == 9, "the number below is written as a word; update both READMEs"
    assert "nine frozen model runs" in _text(TOP_README)
    assert "Nine frozen runs are committed" in _text(THIS_README)


def test_the_rejected_line_count_is_what_this_readme_says():
    frozen = json.load(open(harness.CONFORMANCE, encoding="utf-8"))["per_run"]
    rejected = sum(n for counts in frozen.values() for c, n in counts.items() if c != "ok")
    assert f"{rejected} of these lines do not" in _text(THIS_README), (
        f"the grammar rejects {rejected} recorded lines; this README says otherwise")


def test_the_check_count_is_what_both_documents_say():
    """Counted from this run's own collection, so it cannot be asserted from memory."""
    collected = conftest.COLLECTED[0]
    assert collected, "no tests were collected"
    for path in (TOP_README, THIS_README):
        assert f"{collected} checks" in _text(path), (
            f"{os.path.basename(path)} does not say {collected} checks --- this run collected that many")


def test_the_sibling_mcp_suites_are_counted_as_the_readme_counts_them():
    """The repository README's testing table speaks for suites this gate does not run.

    It said 29 for a long time; there are 26 test functions, and CI runs 22 of them because
    a whole class in the reranking suite is marked `slow` and `integration`. Nothing was
    checking, so counting them here is cheaper than the next person believing the wrong
    number -- and it costs no install, because the count comes from the source.
    """
    total, in_ci = harness.mcp_test_counts()
    text = _text(TOP_README)
    assert f"{total} pytest tests between them" in text, (
        f"the two MCP suites hold {total} test functions; the README says otherwise")
    assert f"{in_ci} of which run in CI" in text, (
        f"CI runs {in_ci} of them (the rest carry a slow or integration marker); "
        "the README says otherwise")
