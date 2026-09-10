"""Every published figure of every frozen run, re-derived from the rows it came from.

This is the gate that makes an evaluation number durable. The run itself cannot be repeated
on a free runner --- it needs a GPU, a checkpoint held outside git and half an hour --- but
the rows it produced are committed, and the arithmetic that turned them into a summary is
cheap and deterministic. So CI does the arithmetic again, with a second implementation, and
fails if the artifact's own headline stops matching its own rows.
"""

import pytest

import harness

BENCH = harness.bench_files()
IDS = [harness.tag(p) for p in BENCH]


@pytest.mark.parametrize("path", BENCH, ids=IDS)
def test_summary_is_what_the_rows_say(path):
    doc = harness.load_bench(path)
    want, got = harness.published(doc), harness.rescore(doc)
    assert set(want) == set(got), "the artifact and the re-scorer disagree on which figures exist"
    differences = {k: {"published": want[k], "re-derived": got[k]} for k in want if want[k] != got[k]}
    assert not differences, differences


@pytest.mark.parametrize("path", BENCH, ids=IDS)
def test_the_scorer_invariants_hold_in_the_rows(path):
    """Two rules the scorer applies, checked against the data it wrote.

    A gold-fingerprint hit is only computed for questions with a canonical derivation, and
    those are executable by definition; an unanswerable question succeeds only by abstaining.
    Rows that break either rule would mean the summary averaged over a set the scorer never
    intended --- exactly the kind of defect a rate hides.
    """
    for row in harness.load_bench(path)["rows"]:
        if row["gold_fp_hit"]:
            assert row["status"] == "EXECUTED", f"{row['id']}: fingerprint hit on a non-executable question"
        if row["status"] != "EXECUTED":
            assert bool(row["success"]) == (row["terminal"] == "pass"), (
                f"{row['id']}: success on an unanswerable question that did not abstain")


@pytest.mark.parametrize("path", BENCH, ids=IDS)
def test_terminals_and_steps_are_consistent(path):
    """Each session ends exactly once, and its recorded step count is its trajectory."""
    for row in harness.load_bench(path)["rows"]:
        assert row["terminal"] in harness.TERMINALS, f"{row['id']}: {row['terminal']}"
        assert row["steps"] == len(row["traj"]), f"{row['id']}: steps ≠ len(traj)"
        assert row["steps"] > 0, f"{row['id']}: an empty trajectory"


def test_every_run_used_the_same_answerable_stratum():
    """The one figure the re-scorer takes on trust, checked for agreement across nine runs.

    Membership of the answerable stratum needs the graph pack, which is exported data and not
    in the repository, so each run's ``n.answerable`` is read rather than re-derived. Nine
    independent runs agreeing on it is the evidence available here: a stratum that moved
    between runs would make the ladder's columns incomparable, which is the failure that
    would actually matter.
    """
    sizes = {harness.tag(p): harness.load_bench(p)["summary"]["n"]["answerable"] for p in BENCH}
    assert len(set(sizes.values())) == 1, f"the ladder's runs scored different strata: {sizes}"
    size = next(iter(sizes.values()))
    for path in BENCH:
        executable = sum(1 for r in harness.load_bench(path)["rows"] if r["status"] == "EXECUTED")
        assert size <= executable, (
            f"{harness.tag(path)}: {size} answerable but only {executable} executable questions")
