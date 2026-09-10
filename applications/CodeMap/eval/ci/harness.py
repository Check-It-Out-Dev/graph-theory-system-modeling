"""Replay evaluation: re-score frozen model runs on a machine with no GPU and no model.

The evaluation of a navigator model has two planes. The OFFLINE plane runs the model --- a
GPU, a 2.5 GB quantised checkpoint, half an hour of wall time per configuration --- and
writes one artifact per run into ``training/data/BENCH_<tag>.json``: every question, the
trajectory the model emitted, and the verdict the scorer gave it. The CI plane never runs a
model. It takes those frozen artifacts and re-derives every published figure from the rows,
with today's scorer, today's DSL grammar and today's question bank.

That split is what makes an evaluation gate possible on a free runner, and it gates the half
that actually rots: a scorer that silently changes its arithmetic, a grammar that quietly
starts accepting what it used to reject, a question bank that drifts away from the corpus the
numbers were measured on, a figure in a document that no artifact supports any more.

One quantity is NOT re-derivable here and the code says so where it is used: the answerable
stratum (which of the 85 executable questions have a canonical reference derivation) needs
the graph pack, which is exported data and not in the repository. Its size is read from the
artifact; every rate computed against it is re-derived.

Usage:
    python harness.py --summary        markdown comparison of every frozen run
    python harness.py --freeze         rewrite conformance.json from today's grammar
"""

import argparse
import ast
import hashlib
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODEMAP = os.path.dirname(os.path.dirname(HERE))
BENCH_DIR = os.path.join(CODEMAP, "training", "data")
BANK = os.path.join(CODEMAP, "eval", "q", "mfq_all.jsonl")
PROMPTS = os.path.join(CODEMAP, "training")
CONFORMANCE = os.path.join(HERE, "conformance.json")
CLAIMS = os.path.join(HERE, "claims.json")

sys.path.insert(0, os.path.join(CODEMAP, "app"))
from dsl import VERBS, ParseError, parse  # noqa: E402

TERMINALS = ("answer", "pass", "stall", "invalid")


# --------------------------------------------------------------------------- loading

def bench_files():
    """Every frozen run artifact, in a stable order."""
    return sorted(
        os.path.join(BENCH_DIR, f)
        for f in os.listdir(BENCH_DIR)
        if f.startswith("BENCH_") and f.endswith(".json")
    )


def load_bench(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_bank():
    """The 104-question bank the runs were scored against."""
    with open(BANK, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def tag(path):
    return os.path.basename(path)[len("BENCH_"):-len(".json")]


# ------------------------------------------------------------------- replay scoring

def rescore(doc):
    """Re-derive every published figure from the rows.

    The formulas are the scorers' own (``app/bench_promptonly.py``, ``app/bench_bigtier.py``);
    this is a second implementation on purpose --- two implementations that agree are evidence,
    one implementation checking itself is not.

    ``n.answerable`` is read from the artifact, not re-derived: membership needs the graph
    pack. Every rate below is divided by it, so a wrong stratum size still shows up as a
    wrong rate.
    """
    rows = doc["rows"]
    n_answerable = doc["summary"]["n"]["answerable"]
    executable = [r for r in rows if r["status"] == "EXECUTED"]
    unanswerable = [r for r in rows if r["status"] != "EXECUTED"]

    out = {
        "n.total": len(rows),
        "n.unanswerable": len(unanswerable),
        "rate.answerable": _rate(sum(bool(r["success"]) for r in executable), n_answerable),
        "rate.gold_fp_hit": _rate(sum(bool(r["gold_fp_hit"]) for r in executable), n_answerable),
        "rate.pass_on_unanswerable": _rate(
            sum(bool(r["success"]) for r in unanswerable), len(unanswerable)),
        "terminals": {t: sum(1 for r in rows if r["terminal"] == t) for t in TERMINALS},
        "steps.mean": round(statistics.mean(r["steps"] for r in rows), 2),
        "steps.p50": statistics.median(r["steps"] for r in rows),
        "seconds_p50": round(statistics.median(r["seconds"] for r in rows), 2),
        "backtrack_rate": _rate(sum(r["backtracks"] for r in rows), len(rows)),
    }

    # the cypher-rung runs carry two further blocks
    answered = [r for r in rows if r.get("content_f1") is not None]
    if answered:
        solid = [r for r in rows if r.get("solid") is not None]
        cypher_steps = sum(r.get("cypher_steps") or 0 for r in rows)
        out.update({
            "n.answered_with_gold": len(answered),
            "content.f1_p50": round(statistics.median(r["content_f1"] for r in answered), 4),
            "content.f1_mean": round(statistics.mean(r["content_f1"] for r in answered), 4),
            "content.number_f1_p50": round(
                statistics.median(r["number_f1"] for r in answered), 4),
            "content.solid_rate": _rate(sum(1 for r in solid if r["solid"]), len(solid)),
            "cypher.sessions_using": _rate(
                sum(1 for r in rows if r.get("cypher_steps")), len(rows)),
            "cypher.steps_total": cypher_steps,
            "cypher.error_rate": _rate(
                sum(r.get("cypher_errors") or 0 for r in rows), cypher_steps),
        })
    return out


def published(doc):
    """The same figures as the artifact states them."""
    s = doc["summary"]
    block = s.get("task_success") or s["route"]
    out = {
        "n.total": s["n"]["total"],
        "n.unanswerable": s["n"]["unanswerable"],
        "rate.answerable": block["answerable"],
        "rate.gold_fp_hit": block["gold_fp_hit"],
        "rate.pass_on_unanswerable": block["pass_on_unanswerable"],
        "terminals": s["terminals"],
        "steps.mean": s["steps"]["mean"],
        "steps.p50": s["steps"]["p50"],
        "seconds_p50": s["seconds_p50"],
        "backtrack_rate": s["backtrack_rate"],
    }
    if "content" in s:
        out.update({
            "n.answered_with_gold": s["n"]["answered_with_gold"],
            "content.f1_p50": s["content"]["f1_p50"],
            "content.f1_mean": s["content"]["f1_mean"],
            "content.number_f1_p50": s["content"]["number_f1_p50"],
            "content.solid_rate": s["content"]["solid_rate"],
            "cypher.sessions_using": s["cypher"]["sessions_using"],
            "cypher.steps_total": s["cypher"]["steps_total"],
            "cypher.error_rate": s["cypher"]["error_rate"],
        })
    return out


def _rate(num, den):
    return round(num / max(1, den), 4)


# -------------------------------------------------------------- grammar conformance

def classify(step):
    """How today's grammar judges one recorded trajectory step.

    ``ok`` or an error CLASS --- never the message, which carries the offending text and
    would make the fingerprint a copy of the corpus.
    """
    try:
        parse(step)
        return "ok"
    except ParseError as exc:
        message = str(exc)
        if message.startswith("not a CMDSL expression"):
            return "not-an-expression"
        if message.startswith("unknown verb"):
            return "unknown-verb"
        return "wrong-arity"


def conformance():
    """Every recorded step of every frozen run, judged by today's grammar.

    The trajectories are 4,936 lines a real model produced under a real prompt, each with a
    verdict the run recorded. They make a conformance suite no hand-written fixture could
    match: loosening the parser to accept a trailing semicolon, or tightening it, re-judges
    lines here and the fingerprint moves.
    """
    per_file, digest = {}, hashlib.sha256()
    for path in bench_files():
        doc = load_bench(path)
        counts = {}
        for row in doc["rows"]:
            for index, step in enumerate(row.get("traj") or []):
                verdict = classify(step)
                counts[verdict] = counts.get(verdict, 0) + 1
                digest.update(f"{tag(path)}\x1f{row['id']}\x1f{index}\x1f{verdict}\x1e"
                              .encode("utf-8"))
        per_file[tag(path)] = dict(sorted(counts.items()))
    return {
        "steps": sum(sum(c.values()) for c in per_file.values()),
        "fingerprint": digest.hexdigest(),
        "per_run": per_file,
    }


def unparseable_examples(limit=10):
    """The recorded steps today's grammar rejects, for a failure message worth reading."""
    out = []
    for path in bench_files():
        for row in load_bench(path)["rows"]:
            for step in row.get("traj") or []:
                verdict = classify(step)
                if verdict != "ok":
                    out.append((tag(path), row["id"], verdict, step[:90]))
                    if len(out) >= limit:
                        return out
    return out


# ------------------------------------------------------------------ prompt contract

def _is_declaration(token):
    verb, sep, raw = token.partition("(")
    return bool(sep) and verb.isalpha() and verb.islower() and raw.endswith(")")


def prompt_verb_table(text):
    """The action space a master prompt declares, as {verb: arity}.

    The prompt writes the arity the way a person reads it --- ``map()``, ``find(term)``,
    ``impact(entity[,depth])``, ``seam(subA,subB)`` --- and this turns that back into the
    parser's own notation so the two can be compared.

    Only DECLARATION lines count: a line whose every token is ``verb(...)``. The prose below
    the list names verbs too ("the runtime already ran cache() for you"), and reading arity
    out of a sentence would make the contract say whatever the sentence happened to phrase.
    """
    table = {}
    for line in text.splitlines():
        tokens = line.split()
        if not tokens or not all(_is_declaration(t) for t in tokens):
            continue
        for token in tokens:
            verb, _, raw = token.partition("(")
            inner = raw[:-1]
            if inner == "":
                table[verb] = 0
                continue
            required = inner.split("[")[0]
            low = len([p for p in required.split(",") if p.strip()])
            high = low + inner.count("[")
            table[verb] = low if low == high else (low, high)
    return table


def master_prompts():
    return sorted(
        os.path.join(PROMPTS, f)
        for f in os.listdir(PROMPTS)
        if f.startswith("master_prompt_") and f.endswith(".txt")
    )


BIG_TIER = os.path.join(CODEMAP, "app", "big_tier.py")


def big_tier_verbs():
    """The verb set the big tier's system prompt offers.

    A different contract from the master prompts: its ACTIONS block writes one call per line
    followed by a plain-English gloss ("optional depth 2nd arg"), so the verb NAMES are
    machine-readable and the arities are not. The test compares the set, not the arity.
    """
    text = open(BIG_TIER, encoding="utf-8").read()
    start = text.index("ACTIONS - respond")
    block = text[start:text.index("NATIVE QUERY:", start)]
    return {m.group(1) for m in re.finditer(r"^\s{2,}([a-z]+)\(", block, re.M)}


# ------------------------------------------------------------- the sibling suites

REPO = os.path.dirname(os.path.dirname(CODEMAP))
MCP_PACKAGES = ("McpServerForEmbeddings", "McpServerForReranking")


def mcp_test_counts():
    """Test functions in the two MCP suites, and how many of them CI actually runs.

    Counted from the source rather than from a run, because this gate installs neither
    package. A `slow` or `integration` marker keeps a test out of CI, and the marker can sit
    on the test OR on the class around it -- the reranking suite marks a whole
    ``TestIntegration`` class, which is why counting decorators on functions alone reports
    four tests that never run.
    """
    total = in_ci = 0
    for package in MCP_PACKAGES:
        tests_dir = os.path.join(REPO, package, "tests")
        for name in sorted(os.listdir(tests_dir)):
            if not name.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(tests_dir, name), encoding="utf-8").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    excluded = _excluded(node)
                    for child in node.body:
                        if _is_test(child):
                            total += 1
                            in_ci += not (excluded or _excluded(child))
                elif _is_test(node) and _toplevel(tree, node):
                    total += 1
                    in_ci += not _excluded(node)
    return total, in_ci


def _is_test(node):
    return (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_"))


def _excluded(node):
    return any(m in ast.unparse(d) for d in node.decorator_list
               for m in ("mark.slow", "mark.integration"))


def _toplevel(tree, node):
    return node in tree.body


# ------------------------------------------------------------------------- reporting

def comparison_table():
    """Nine frozen runs, two models, four prompt configurations --- side by side.

    This is the reason the gate is worth looking at rather than only worth passing: the
    configurations were an experiment, and the experiment's result is re-derived on every
    run instead of being remembered.
    """
    lines = [
        "| run | terminals a/p/s/i | answerable | gold fp | pass on unanswerable | steps p50 | s p50 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for path in bench_files():
        got = rescore(load_bench(path))
        t = got["terminals"]
        lines.append(
            f"| {tag(path)} | {t['answer']}/{t['pass']}/{t['stall']}/{t['invalid']} "
            f"| {got['rate.answerable']:.4f} | {got['rate.gold_fp_hit']:.4f} "
            f"| {got['rate.pass_on_unanswerable']:.4f} | {got['steps.p50']:g} "
            f"| {got['seconds_p50']:g} |"
        )
    return "\n".join(lines)


def best_by(metric):
    """Which frozen configuration leads on one metric, re-derived rather than recalled."""
    scored = [(rescore(load_bench(p))[metric], tag(p)) for p in bench_files()]
    return max(scored)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", action="store_true", help="markdown comparison of the runs")
    ap.add_argument("--freeze", action="store_true", help="rewrite conformance.json")
    args = ap.parse_args()

    if args.freeze:
        with open(CONFORMANCE, "w", encoding="utf-8") as fh:
            json.dump(conformance(), fh, indent=1)
            fh.write("\n")
        print("wrote", CONFORMANCE)

    if args.summary:
        con = conformance()
        print("### CodeMap evaluation replay\n")
        print(f"{len(bench_files())} frozen runs re-scored, {con['steps']} recorded DSL steps "
              f"re-judged by today's grammar.\n")
        print(comparison_table())
        print()
        for metric, label in (("rate.answerable", "task success on answerable"),
                              ("rate.gold_fp_hit", "gold fingerprint hit"),
                              ("rate.pass_on_unanswerable", "abstention on unanswerable")):
            value, name = best_by(metric)
            print(f"- best {label}: **{name}** at {value:.4f}")


if __name__ == "__main__":
    main()
