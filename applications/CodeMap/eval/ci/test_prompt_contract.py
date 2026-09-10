"""The prompt is a contract with the executor, so it is tested like one.

A navigator model's whole action space is what its system prompt says it may emit. If the
prompt and the parser disagree by one verb or one argument, the model spends its turns being
told "not a valid action" and every metric downstream moves for a reason no metric names.
The training pipeline states this as law --- the master prompt and the DSL version are frozen
together before a single pair is generated --- and these tests are that law, executable.
"""

import pytest

import harness


def _normalise(table):
    return {verb: tuple(arity) if isinstance(arity, (tuple, list)) else arity
            for verb, arity in table.items()}


@pytest.mark.parametrize("path", harness.master_prompts(), ids=harness.os.path.basename)
def test_master_prompt_declares_exactly_the_parser_verbs(path):
    """Same verbs, same arities --- in both directions.

    A verb in the prompt the parser rejects is a turn the model cannot spend; a verb in the
    parser no prompt offers is dead surface the model will never reach.
    """
    declared = _normalise(harness.prompt_verb_table(open(path, encoding="utf-8").read()))
    implemented = _normalise(harness.VERBS)
    assert declared == implemented, (
        f"{harness.os.path.basename(path)} and app/dsl.py disagree: "
        f"{ {v: (declared.get(v), implemented.get(v)) for v in set(declared) | set(implemented) if declared.get(v) != implemented.get(v)} }"
    )


@pytest.mark.parametrize("path", harness.master_prompts(), ids=harness.os.path.basename)
def test_every_declared_call_shape_parses(path):
    """The prompt's own example shapes must survive the parser.

    Not a restatement of the arity test: this builds a call for each verb at each legal arity
    and pushes it through ``parse``, so a change to quoting, splitting or the expression
    regex is caught even when the verb table still agrees.
    """
    for verb, arity in harness.prompt_verb_table(open(path, encoding="utf-8").read()).items():
        low, high = (arity, arity) if isinstance(arity, int) else tuple(arity)
        for count in range(low, high + 1):
            call = f"{verb}({', '.join(f'a{i}' for i in range(count))})"
            got_verb, got_args = harness.parse(call)
            assert got_verb == verb and len(got_args) == count, call


def test_the_grammar_rejects_what_is_not_in_the_contract():
    """The parser is a gate, not a suggestion --- so prove it still refuses.

    Every trajectory verdict in the frozen runs, and the ``invalid`` terminal itself, rests on
    these refusals. A parser that stopped refusing would turn recorded failures into recorded
    successes without anyone editing a number.
    """
    for expression in ("cypher(MATCH (n) RETURN n)",   # the big tier's verb, not CMDSL's
                       "find(a, b)",                    # arity
                       "answer()",                      # arity
                       "find(Foo.java);",               # the trailing semicolon that cost 17 answers
                       "Let me find(Foo.java)"):        # narration
        with pytest.raises(harness.ParseError):
            harness.parse(expression)


def test_the_big_tier_extends_the_action_space_by_exactly_one_verb():
    """The large local tier runs a second contract, and the difference is the point.

    Its prompt drops ``cache`` (the runtime pre-runs it) and adds ``cypher``, read-only
    openCypher against the pack's database --- the shapes no verb covers. Anything else
    appearing here would be a verb offered to a model with nothing behind it.
    """
    declared = harness.big_tier_verbs()
    core = set(harness.VERBS)
    assert declared - core == {"cypher"}, f"unexpected verbs in the big tier: {declared - core}"
    assert core - declared == {"cache"}, f"unexpectedly dropped: {core - declared}"

    source = open(harness.BIG_TIER, encoding="utf-8").read()
    assert "_CYPHER_RX" in source and "def run_cypher" in source, (
        "the big tier offers cypher() but no longer implements it")
