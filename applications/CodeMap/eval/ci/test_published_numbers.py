"""Numbers in the prose, checked against the artifacts they claim to come from.

The frontend repository gates its published figures this way and it caught eleven stale ones
on the first run. Evaluation numbers rot the same way and worse: a table of prompt rungs is
copied into a README, a run is repeated, the artifact moves and the table does not.

A claim is gated when an artifact supports it: the literal text must still be in the document,
and the artifact re-scored today must still round to the claimed value at the claim's own
precision. A claim is ungated when no artifact in this repository supports it --- the trained
navigator's own gate table, for instance, whose test snapshot and checkpoint live outside git.
Ungated claims are declared, with the reason, rather than left unmentioned.
"""

import json

import pytest

import harness

CLAIMS = json.load(open(harness.CLAIMS, encoding="utf-8"))
SCORED = {harness.tag(p): harness.rescore(harness.load_bench(p)) for p in harness.bench_files()}


def _doc(relative):
    with open(harness.os.path.join(harness.CODEMAP, relative), encoding="utf-8") as fh:
        return fh.read()


def _label(claim):
    return f"{claim['doc'].split('/')[-1]}::{claim.get('metric', 'ungated')}={claim.get('value', '')}"


def _lookup(scored, metric):
    head, _, tail = metric.partition(".")
    if metric in scored:
        return scored[metric]
    if head == "terminals":
        return scored["terminals"][tail]
    raise KeyError(metric)


@pytest.mark.parametrize("claim", CLAIMS["gated"], ids=_label)
def test_the_document_still_says_it(claim):
    assert claim["text"] in _doc(claim["doc"]), (
        f"{claim['doc']} no longer contains {claim['text']!r} --- the manifest is stale, or a "
        "published figure was edited without re-gating it")


@pytest.mark.parametrize("claim", CLAIMS["gated"], ids=_label)
def test_the_artifact_still_supports_it(claim):
    scored = SCORED[claim["run"]]
    actual = _lookup(scored, claim["metric"])
    claimed = claim["value"]
    if isinstance(claimed, float):
        decimals = len(str(claimed).split(".")[1])
        actual = round(actual, decimals)
    assert actual == claimed, (
        f"{claim['doc']} claims {claim['metric']} = {claimed} for {claim['run']}; the artifact, "
        f"re-scored, gives {actual}")


@pytest.mark.parametrize("claim", CLAIMS["ungated"], ids=_label)
def test_an_ungated_claim_is_still_in_the_document_and_still_has_a_reason(claim):
    """An ungated number is a promise about what is missing, so it is kept honest too.

    If the text goes, the entry should go with it; if the reason goes, nobody can tell whether
    the number was ever backed by anything.
    """
    assert claim["text"] in _doc(claim["doc"]), (
        f"{claim['doc']} no longer contains {claim['text']!r} --- drop the entry from claims.json")
    assert len(claim["reason"]) > 40, "state why no artifact backs this number"


def test_every_frozen_run_is_either_cited_or_declared_uncited():
    """A run in the repository is either backing a published number or explicitly not.

    Without this the manifest could quietly narrow to the runs that still agree: drop the
    claims that broke, and the gate goes green while the document keeps saying them.
    """
    cited = {claim["run"] for claim in CLAIMS["gated"]}
    declared = {entry["run"] for entry in CLAIMS["uncited"]}
    assert cited <= set(SCORED), f"the manifest cites runs that are not here: {cited - set(SCORED)}"
    assert not (cited & declared), f"both cited and declared uncited: {cited & declared}"
    missing = set(SCORED) - cited - declared
    assert not missing, (
        f"frozen runs neither cited by a document nor declared uncited: {sorted(missing)} --- "
        "add them to claims.json, either as a gated claim or with a reason")
    for entry in CLAIMS["uncited"]:
        assert len(entry["reason"]) > 40, f"{entry['run']}: state why nothing cites it"
