# CodeMap loop smoke test — proves the engine + DSL end-to-end with NO model:
# scripted DSL sessions over real gold questions. This is rung 0.5 of the ladder
# (the machinery floor beneath rung 1's bare model).
#
# Usage: PYTHONUTF8=1 python smoke.py

import sys
from engine import Engine
from dsl import execute, parse, emit_gbnf, ParseError


def main():
    e = Engine()
    print(f"pack loaded: {len(e.ents)} entities, {len(e.edges)} edges, "
          f"{len(e.l2)} L2, {len(e.mfq)} MFQ ({len(e.invalidated)} invalidated), "
          f"ladybug={'ON' if e.lb else 'OFF'}")

    # 1. cache tier: a NON-invalidated gold hits by alias with the gold answer
    r = execute(e, 'cache("Valid JWT yet forbidden")')
    assert r["kind"] == "cache_hit" and "403" in r["answer"], r
    print(f"1 cache: HIT {r['id']} score {r['score']}")

    # 1b. an INVALIDATED gold must MISS even on an exact alias — the delta invalidation
    # layer refusing to serve stale answers (M03 depends on sub-3, changed 2026-09-02)
    r = execute(e, 'cache("What is the blast radius of UserRepository?")')
    assert r["kind"] == "cache_miss", r
    print(f"1b invalidation: exact alias of invalidated gold correctly MISSES (score {r['score']})")

    # 2. cache miss on a novel phrasing falls through to traversal
    r = execute(e, 'cache("what colour is the bikeshed")')
    assert r["kind"] == "cache_miss"
    print(f"2 cache miss -> affordance {r['affordances']}")

    # 3. the mandated descent: map -> enter -> spine
    r = execute(e, "map()")
    assert r["kind"] == "l1" and r["affordances"]
    r = execute(e, "enter(11)")
    assert r["kind"] == "l2" and "Subscriptions" in (r["name"] or ""), r.get("name")
    r = execute(e, "spine(11)")
    assert r["kind"] == "spine" and r["spines"], r
    print(f"3 descent: L1 -> L2({r['sub']}) -> spine ({len(r['spines'])} spines)")

    # 4. impact on Ladybug-backed find + memory impact — cross-check the gold number
    r = execute(e, "find(UserRepository)")
    assert any(h["name"] == "UserRepository.java" for h in r["hits"])
    r = execute(e, "impact(UserRepository.java)")
    # 79 distinct FILES (184 typed rows; 141 is the external in-EDGE count) — this very
    # assertion caught a mislabel in the hand-authored gold answer; corrected at source.
    assert r["dependents"] == 79, f"expected 79 distinct dependents, got {r['dependents']}"
    print(f"4 impact: {r['dependents']} distinct dependents across {len(r['subsystems'])} subsystems (gold M03 cross-checked)")

    # 5. seam + cohort + health + read + terminals
    r = execute(e, "seam(11, 10)")
    assert r["edges"] >= 30, r["edges"]
    r = execute(e, "cohort(address.service.ts)")
    assert r["cohorts"] and r["cohorts"][0]["hub"] == "address.service.ts"
    r = execute(e, "health(coupling)")
    assert r["rows"][0]["external_ratio"] >= 0.99
    r = execute(e, "read(PaymentsDisabledBootGuard.java)")
    assert "checkItOut-be2" in r["path"]
    r = execute(e, 'answer("done")')
    assert r["done"]
    r = execute(e, 'pass("beyond v0")')
    assert r["done"] and r["kind"] == "pass"
    print("5 seam/cohort/health/read/answer/pass: all OK")

    # 6. parser rejects garbage helpfully; grammar emits
    for bad in ["drop table", "impact()", "enter(1,2,3)", "raw(x)"]:
        try:
            parse(bad)
            raise AssertionError(f"parser accepted {bad!r}")
        except ParseError:
            pass
    assert "root ::=" in emit_gbnf()
    print("6 parser rejections + GBNF emit: OK")
    print("SMOKE PASS — the loop machinery works with zero model attached")


if __name__ == "__main__":
    sys.exit(main())
