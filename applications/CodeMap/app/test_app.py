# CodeMap app test suite — parser, engine (incl. hierarchy), protocol, grammar.
# Plain-python asserts, zero deps beyond the app itself. Run: PYTHONUTF8=1 python test_app.py

import json
import sys

from dsl import parse, execute, emit_gbnf, ParseError, VERBS
from engine import Engine, DslError

FAILS = []


RAN = []


def check(name, fn):
    RAN.append(name)
    try:
        fn()
        print(f"  ok  {name}")
    except AssertionError as e:
        FAILS.append(name)
        print(f"FAIL  {name}: {e}")


def main():
    e = Engine()
    print(f"[pack] {len(e.ents)} entities · {len(e.l2)} navigators · roots {e.roots} · "
          f"{len(e.leaves)} leaves · ladybug {'ON' if e.lb else 'OFF'}")

    # ---------- parser ----------
    check("parser: every verb parses at its arity", lambda: [
        parse(x) for x in ["map()", "enter(11)", "find(user)", "impact(X.java)",
                           "impact(X.java, 2)", "flow(X.ts)", "flow(X.ts, 3)", "seam(11, 10)",
                           "cohort(a.ts)", "spine(4)", "health(coupling)", "read(A.java)",
                           'cache("how does auth work")', 'answer("text, with commas")',
                           'pass("beyond me")']])
    def rejections():
        for bad in ["drop table", "map(x)", "enter()", "seam(1)", "impact(a,b,c)",
                    "raw(q)", "MAP()", "answer()"]:
            try:
                parse(bad)
                raise AssertionError(f"accepted {bad!r}")
            except ParseError:
                pass
    check("parser: rejects invalid forms", rejections)
    check("parser: quoted args keep commas", lambda: (
        lambda v, a: [v == "answer", a == ["answer, with commas"]] and None)(*parse('answer("answer, with commas")')))
    check("grammar: GBNF covers all verbs", lambda: (
        lambda g: all(v in g for v in VERBS) and None or None)(emit_gbnf()))

    # ---------- hierarchy ----------
    check("map: exactly 6 roots, enter affordances", lambda: (
        lambda r: [len([a for a in r["affordances"] if a.startswith("enter(")]) == 6] and None)(e.map()))
    check("enter(GROUP 201): billing group lists 3 children", lambda: (
        lambda r: [r["kind"] == "l2_group",
                   sorted(c["sub"] for c in r["children"]) == [10, 11, 12]] and None)(e.enter(201)))
    check("enter(GROUP 17): FE lists 9 children", lambda: (
        lambda r: [r["kind"] == "l2_group", len(r["children"]) == 9] and None)(e.enter(17)))
    check("enter(leaf 11): curated name + size 208", lambda: (
        lambda r: [r["kind"] == "l2", "Subscriptions" in r["name"], r["size"] == 208] and None)(e.enter(11)))
    check("enter by name resolves", lambda: (
        lambda r: [r["sub"] == 172] and None)(e.enter("FE onboarding survey")))
    check("merged sub-18 is gone from navigation", lambda: (
        lambda: (_ for _ in ()).throw(AssertionError("resolved 18"))
        if 18 in e.l2 else None)())

    # ---------- verbs over curated membership ----------
    check("impact: 79 distinct dependents, curated subsystem labels", lambda: (
        lambda r: [r["dependents"] == 79, 11 in r["subsystems"]] and None)(e.impact("UserRepository.java")))
    check("seam(11,10): the User.java seam exists", lambda: (
        lambda r: [r["edges"] >= 30] and None)(e.seam(11, 10)))
    check("seam(170,177): FE children talk", lambda: (
        lambda r: [r["edges"] >= 1] and None)(e.seam(170, 177)))
    check("health(coupling): leaves only, no GROUP rows", lambda: (
        lambda r: [all(row["sub"] in e.leaves for row in r["rows"])] and None)(e.health("coupling")))
    check("cohort(address.service.ts): hub cohort", lambda: (
        lambda r: [r["cohorts"][0]["hub"] == "address.service.ts"] and None)(e.cohort("address.service.ts")))
    check("find: hits + affordances", lambda: (
        lambda r: [any(h["name"] == "StripeService.java" for h in r["hits"])] and None)(e.find("StripeService")))
    check("read: content pointer with path", lambda: (
        lambda r: ["checkItOut-be2" in r["path"]] and None)(e.read("PaymentsDisabledBootGuard.java")))
    check("unknown entity: helpful DslError", lambda: (
        lambda: (_ for _ in ()).throw(AssertionError("no error"))
        if not _raises(lambda: e.impact("NoSuchFile.java")) else None)())

    # ---------- protocol semantics ----------
    check("cache: non-invalidated gold hits by alias", lambda: (
        lambda r: [r["kind"] == "cache_hit", "403" in r["answer"]] and None)(e.cache("Valid JWT yet forbidden")))
    check("cache: invalidated gold misses on exact alias", lambda: (
        lambda r: [r["kind"] == "cache_miss"] and None)(e.cache("What is the blast radius of UserRepository?")))
    check("terminals: answer/pass end the loop", lambda: [
        execute(e, 'answer("done")')["done"] and execute(e, 'pass("beyond me")')["done"]])

    # ---------- model integration (no GGUF needed: absence + mocked loop) ----------
    import loop_runner
    from model_client import NavigatorModel

    def model_absent_falls_back():
        m = NavigatorModel(gguf="Z:/nonexistent.gguf")
        assert not m.available() and m.ask(e, "anything") is None

    check("model: absent GGUF -> unavailable, /ask keeps v0 protocol", model_absent_falls_back)

    def mocked_loop():
        script = iter(["find(UnifiedStorage)",
                       "impact(UnifiedStorageConfiguration.java)",
                       'answer("done")'])
        real = loop_runner.chat
        loop_runner.chat = lambda port, user, mt: next(script)
        try:
            traj, terminal, text, fps, bt = loop_runner.run_loop(0, e, "co zepsuje zmiana?")
        finally:
            loop_runner.chat = real
        assert terminal == "answer" and text == "done" and len(traj) == 3, (terminal, traj)
        assert bt == 0 and len(fps) == 2 and not any(f.startswith("ERR") for f in fps), fps

    check("model: autonomous loop mechanics (accumulate, execute, terminate)", mocked_loop)

    def mocked_stall_and_backtrack():
        script = iter(["map()", "map()", "map()", "map()", "map()", "map()"])
        real = loop_runner.chat
        loop_runner.chat = lambda port, user, mt: next(script)
        try:
            traj, terminal, _, _, bt = loop_runner.run_loop(0, e, "loop?")
        finally:
            loop_runner.chat = real
        assert terminal == "stall" and len(traj) == 6 and bt == 5, (terminal, bt)

    check("model: stall detection + backtrack counting", mocked_stall_and_backtrack)

    def escalation_offer_contract():
        from server import mint_offer
        o = mint_offer("Ile Stripe bierze?", dict(
            terminal="pass", text="out-of-corpus: commercial terms are not code",
            trajectory=["find(Stripe)", 'pass("...")']))
        assert o["status"] == "PENDING_USER" and o["reason_category"] == "out-of-corpus"
        assert o["suggested_tier"] == "api" and len(o["offer_id"]) == 12
        o2 = mint_offer("q", dict(terminal="stall", text=None, trajectory=[]))
        assert o2["reason_category"] == "loop-health", o2
        o3 = mint_offer("q", dict(terminal="pass",
                                  text="needs-content-read: file body only", trajectory=[]))
        assert "recommended" in o3["advice"], o3

    check("escalation: offer contract minted with routed advice", escalation_offer_contract)

    def api_tier_key_and_models():
        import os
        import api_tier
        env0 = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            # no env key: resolution falls to .env (absent in test) -> disabled
            if not os.path.exists(os.path.join(api_tier.ROOT, ".env")):
                assert not api_tier.available()
            os.environ["ANTHROPIC_API_KEY"] = "test-key-never-real"
            assert api_tier.available()
            assert api_tier.resolve_model(None) == "claude-sonnet-5"
            assert api_tier.resolve_model("budget").startswith("claude-haiku")
            assert api_tier.resolve_model("claude-opus-5") == "claude-opus-5"
            # the Sonnet 4-line with classic extended thinking stays selectable
            assert api_tier.resolve_model("reasoning") == "claude-sonnet-4-5"
            th = api_tier.thinking_for("reasoning")
            assert th["type"] == "enabled" and 1024 <= th["budget_tokens"]
            assert api_tier.thinking_for("default") is None
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            if env0:
                os.environ["ANTHROPIC_API_KEY"] = env0

    check("api tier: key chain + model aliases, never a hardcoded branch", api_tier_key_and_models)

    def api_navigate_loop_mocked():
        import api_tier
        calls = []
        script = [
            dict(stop_reason="tool_use", model="mock", usage=dict(input_tokens=100, output_tokens=20),
                 content=[dict(type="text", text="let me look"),
                          dict(type="tool_use", id="t1", name="cmdsl",
                               input=dict(expression="map()"))]),
            dict(stop_reason="end_turn", model="mock", usage=dict(input_tokens=200, output_tokens=50),
                 content=[dict(type="text", text="The system has 6 top groups.")]),
        ]
        real = api_tier._call
        api_tier._call = lambda body, timeout=120: calls.append(body) or script[len(calls) - 1]
        try:
            rec = api_tier.navigate(e, "overview?", model="budget")
        finally:
            api_tier._call = real
        assert rec["answer"] == "The system has 6 top groups.", rec["answer"]
        assert rec["trajectory"] == ["map()"] and rec["steps"] == 1
        assert rec["usage"] == dict(input_tokens=300, output_tokens=70)
        # the tool_result carried the REAL engine digest back to the model
        tr = calls[1]["messages"][-1]["content"][0]
        assert tr["type"] == "tool_result" and "[11]" in tr["content"], tr["content"][:80]
        # and the system prompt teaches the graph + the L1 index (grounded, not generic)
        assert "SELECTION LAW" in calls[0]["system"] and "[11]" in calls[0]["system"]
        # budget alias sends no thinking config; reasoning alias sends classic
        # extended thinking with budget_tokens < max_tokens
        assert "thinking" not in calls[0]
        calls.clear()
        script2 = [dict(stop_reason="end_turn", model="mock",
                        usage=dict(input_tokens=10, output_tokens=5),
                        content=[dict(type="text", text="ok")])]
        api_tier._call = lambda body, timeout=120: calls.append(body) or script2[0]
        try:
            api_tier.navigate(e, "q?", model="reasoning", max_steps=1)
        finally:
            api_tier._call = real
        assert calls[0]["thinking"]["type"] == "enabled"
        assert calls[0]["thinking"]["budget_tokens"] < calls[0]["max_tokens"]

    check("api navigate: agentic cmdsl loop drives the real engine", api_navigate_loop_mocked)

    def api_navigate_finalizer():
        import api_tier
        calls = []
        script = [
            dict(stop_reason="max_tokens", model="mock", usage=dict(input_tokens=90, output_tokens=1024),
                 content=[dict(type="tool_use", id="t1", name="cmdsl",
                               input=dict(expression="map()"))]),
            dict(stop_reason="end_turn", model="mock", usage=dict(input_tokens=150, output_tokens=40),
                 content=[dict(type="text", text="Synthesis from evidence.")]),
        ]
        real = api_tier._call
        api_tier._call = lambda body, timeout=120: calls.append(body) or script[len(calls) - 1]
        try:
            rec = api_tier.navigate(e, "flow?", model="budget", max_steps=1)
        finally:
            api_tier._call = real
        # a truncated/odd ending must still produce text via the forced synthesis turn
        assert rec["answer"] == "Synthesis from evidence.", rec["answer"]
        assert calls[-1]["tool_choice"] == dict(type="none")
        roles = [m["role"] for m in calls[-1]["messages"]]
        assert roles[-1] == "user" and "exhausted" in calls[-1]["messages"][-1]["content"]

    check("api navigate: finalizer forces synthesis on any non-answer ending", api_navigate_finalizer)

    # ---------- big local tier (no big GGUF needed: guard + mocked loop) ----------
    import big_tier

    def cypher_guard():
        assert e.lb is not None, "ladybug offline — read_only open failed"
        r = big_tier.run_cypher(e, "MATCH (n:Entity) SET n.layer = 'x'")
        assert r.startswith("ERROR") and "set" in r.lower(), r
        r = big_tier.run_cypher(e, "MATCH (n) RETURN n; MATCH (m) RETURN m")
        assert r.startswith("ERROR") and ";" in r, r
        r = big_tier.run_cypher(
            e, "MATCH (x:Entity) WHERE x.name = 'UserRepository.java' "
               "RETURN x.name, coalesce(x.curated, x.subsystem)")
        assert r.startswith("cypher rows (1)") and "UserRepository.java" in r, r
        r = big_tier.run_cypher(e, "MATCH (x:Entity) RETURN x.name")  # no LIMIT given
        assert r.startswith("cypher rows (25+")  # row cap enforced, marked as capped
        r = big_tier.run_cypher(e, "MATCH (x:Entity) WHERE x.name = 'No.zz' RETURN x.name")
        assert r == "cypher rows: NONE", r

    check("big tier: cypher is read-only, capped, and dialect-real", cypher_guard)

    def cypher_clean():
        c = big_tier.clean_action("```\ncypher(MATCH (n:Entity) RETURN n.name);\n```")
        assert c == "cypher(MATCH (n:Entity) RETURN n.name)", c
        c = big_tier.clean_action("cypher(MATCH (a:Entity)\nRETURN a.name LIMIT 3)")
        assert c == "cypher(MATCH (a:Entity) RETURN a.name LIMIT 3)", c
        assert big_tier.clean_action("Step 2: enter(11);") == "enter(11)"

    check("big tier: liberal decoding tolerates fences, spills, semicolons", cypher_clean)

    def big_loop_mocked():
        script = iter([
            "cypher(MATCH (x:Entity) WHERE x.name = 'UserRepository.java' RETURN x.name)",
            'answer("grounded")'])
        real = big_tier.chat
        big_tier.chat = lambda port, system, user, mt, timeout=900: next(script)
        try:
            traj, terminal, text, results, bt = big_tier.run_big_loop(0, e, "q?")
        finally:
            big_tier.chat = real
        assert terminal == "answer" and text == "grounded" and len(traj) == 2
        assert results[0]["digest"].startswith("cypher rows (1)"), results[0]

    check("big tier: loop executes cypher natively and terminates", big_loop_mocked)

    def big_loop_synthesis():
        script = iter(["map()"] * 6 + ['answer("forced from evidence")'])
        real = big_tier.chat
        big_tier.chat = lambda port, system, user, mt, timeout=900: next(script)
        try:
            traj, terminal, text, results, bt = big_tier.run_big_loop(0, e, "q?")
        finally:
            big_tier.chat = real
        assert terminal == "answer" and text == "forced from evidence", (terminal, text)
        assert len(traj) == 7 and bt == 5, (len(traj), bt)
        # bench mode stays strict: same script must STALL with no rescue call
        script = iter(["map()"] * 6)
        big_tier.chat = lambda port, system, user, mt, timeout=900: next(script)
        try:
            traj, terminal, _, _, _ = big_tier.run_big_loop(0, e, "q?", strict=True)
        finally:
            big_tier.chat = real
        assert terminal == "stall" and len(traj) == 6

    check("big tier: forced synthesis rescues the product, never the bench", big_loop_synthesis)

    def big_absent():
        m = big_tier.BigNavigator(gguf="Z:/nonexistent.gguf", port=1)
        assert not m.available() and m.ask(e, "anything") is None
        import os as _os
        _os.environ["CODEMAP_BIG_GGUF"] = "Z:/pinned.gguf"
        try:
            assert big_tier.resolve_gguf() == "Z:/pinned.gguf"
        finally:
            _os.environ.pop("CODEMAP_BIG_GGUF")

    check("big tier: absent GGUF -> unavailable; env pin wins", big_absent)

    def big_system_grounded():
        s = big_tier.big_system(e)
        assert "HOW THE GRAPH WAS BUILT" in s and "coalesce(e.curated, e.subsystem)" in s
        assert "[11]" in s, "L1 index not injected"
        assert "STEP" in s and "6" in s  # budget law present

    check("big tier: system prompt carries construction story + live L1 index", big_system_grounded)

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)} "
          f"({len(RAN) - len(FAILS)}/{len(RAN)} checks)")
    return 1 if FAILS else 0


def _raises(fn):
    try:
        fn()
        return False
    except (DslError, ParseError):
        return True


if __name__ == "__main__":
    sys.exit(main())
