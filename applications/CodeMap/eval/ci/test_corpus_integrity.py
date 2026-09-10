"""The nine runs and the committed question bank have to be the same corpus.

A benchmark number means "this model, on these questions". The questions are in
``eval/q/mfq_all.jsonl``; the runs are in ``training/data/``. Nothing in either file forces
them to agree, and a question quietly edited, re-classified or renumbered would leave every
published rate intact and every published rate wrong.
"""

import pytest

import harness

BANK = harness.load_bank()
BY_ID = {row["id"]: row for row in BANK}
BENCH = harness.bench_files()
IDS = [harness.tag(p) for p in BENCH]


def test_the_bank_is_well_formed():
    assert len(BANK) == 104, f"the ladder is a 104-question bank; found {len(BANK)}"
    assert len(BY_ID) == len(BANK), "duplicate question ids"
    surfaces = [row["q"].strip() for row in BANK]
    assert len(set(surfaces)) == len(surfaces), "two questions share a surface"
    for row in BANK:
        for field in ("id", "q", "gold_status", "archetype", "stratum"):
            assert row.get(field), f"{row.get('id')}: missing {field}"
        assert row["gold_status"] in ("EXECUTED", "CONTENT_POINTER", "COVERAGE_GAP"), row["id"]


@pytest.mark.parametrize("path", BENCH, ids=IDS)
def test_the_run_scored_this_bank_question_for_question(path):
    rows = harness.load_bench(path)["rows"]
    assert {r["id"] for r in rows} == set(BY_ID), (
        f"{harness.tag(path)} was scored against a different question set")
    for row in rows:
        assert row["status"] == BY_ID[row["id"]]["gold_status"], (
            f"{row['id']}: the run recorded {row['status']}, the bank says "
            f"{BY_ID[row['id']]['gold_status']} --- the question was re-classified after the run")


def test_the_unanswerable_stratum_is_the_bank_s_own():
    """Nineteen questions the graph is not expected to answer, and the runs agree on all of them.

    This stratum is the whole abstention metric. It is derivable from committed data alone ---
    a question is unanswerable exactly when the bank does not mark it executable --- so unlike
    the answerable stratum it is re-derived here rather than trusted.
    """
    expected = {row["id"] for row in BANK if row["gold_status"] != "EXECUTED"}
    assert len(expected) == 19, f"the bank's unanswerable stratum moved: {len(expected)}"
    for path in BENCH:
        doc = harness.load_bench(path)
        recorded = {r["id"] for r in doc["rows"] if r["status"] != "EXECUTED"}
        assert recorded == expected, f"{harness.tag(path)}: {recorded ^ expected}"
        assert doc["summary"]["n"]["unanswerable"] == 19, harness.tag(path)


def test_every_gold_answer_that_is_a_dsl_line_still_parses():
    """Where the bank pins a canonical route, that route must remain expressible.

    The bank is data, but the pieces of it that are DSL are contract too: a gold the parser
    can no longer read is a reference the harness can no longer execute.
    """
    checked = 0
    for row in BANK:
        gold = (row.get("gold_answer") or "").strip()
        if gold.startswith(tuple(f"{v}(" for v in harness.VERBS)) and gold.endswith(")"):
            harness.parse(gold)
            checked += 1
    assert checked >= 0  # the bank may hold prose golds only; the assertion is that none break
