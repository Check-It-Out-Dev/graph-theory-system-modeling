"""Deterministic judge and humans fixtures for the quality gate, derived from events.sample.jsonl
(seed 42), plus the expected.json snapshot. Regenerate together when the schema changes."""

import json
import os
import random
import sys

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(R, "eval", "quality"))
FIX = os.path.join(R, "telemetry", "fixtures")


def main():
    rnd = random.Random(42)
    events = [json.loads(l) for l in open(os.path.join(FIX, "events.sample.jsonl"), encoding="utf-8") if l.strip()]
    rows = []
    for e in events:
        if e["event_type"] != "ask" or not e["tier"].startswith("nav-"):
            continue
        good = rnd.random() < 0.8
        abstain = e["terminal"] == "abstain"
        judge = {"located": rnd.choice([4, 5]) if good else rnd.choice([1, 2, 3]),
                 "grounded": rnd.choice([4, 5]) if good else rnd.choice([1, 2, 3]),
                 "correct": rnd.choice([4, 5]) if good else rnd.choice([1, 2, 3]),
                 "abstain": rnd.choice([4, 5]) if (good or abstain) else rnd.choice([2, 3]),
                 "helpful": rnd.choice([3, 4, 5]) if good else rnd.choice([1, 2]), "rationale": "fixture"}
        has = rnd.random() < 0.7
        success = (judge["located"] >= 4) if rnd.random() < 0.9 else (judge["located"] < 4)
        rows.append({"id": e["request_id"], "request_id": e["request_id"], "user": e["user"], "tier": e["tier"], "q": e["q"],
                     "terminal": e["terminal"], "judge": judge, "oracle": {"has": has, "success": success if has else None},
                     "rr_equiv": round(rnd.uniform(0.6, 0.99), 3) if judge["correct"] >= 4 else round(rnd.uniform(0.0, 0.4), 3),
                     "disputed": False})
    for r in rows:
        o = r["oracle"]
        r["disputed"] = bool(o["has"] and ((r["judge"]["located"] >= 4) != bool(o["success"])))
    doc = {"schema": 1, "source": "events.sample.jsonl", "rows": rows,
           "calibration": {"kappa_oracle": 0.71, "n_oracle": sum(1 for r in rows if r["oracle"]["has"]), "kappa_human": 0.64,
                           "n_human": 20, "agreement_rr": 0.9, "n_rr": len(rows), "anchors": {"kappa_now": 0.7, "drift": False},
                           "gate": 0.6, "calibrated": True, "verdict": "calibrated"}}
    with open(os.path.join(FIX, "judge.sample.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    humans = []
    for persona, seed in (("haiku-pm", "BE01"), ("sonnet-newcomer", "FE03"), ("opus-architect", "GR20")):
        for mode in ("codemap", "baseline"):
            if persona == "opus-architect" and mode == "baseline":
                continue
            humans.append({"night": "2025-09-16", "persona": persona, "mode": mode, "seed_id": seed,
                           "usage": {"input_tokens": 100, "output_tokens": 3000 if mode == "codemap" else 6000,
                                     "cache_read_input_tokens": 600000 if mode == "codemap" else 900000, "cache_creation_input_tokens": 60000},
                           "num_turns": 12 if mode == "codemap" else 20, "duration_ms": 110000 if mode == "codemap" else 260000,
                           "is_error": False,
                           "report": {"would_have_found_alone": "partly", "minutes_saved_estimate": 15} if mode == "codemap" else None})
    with open(os.path.join(FIX, "humans.sample.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for h in humans:
            f.write(json.dumps(h, sort_keys=True) + "\n")
    import quality
    out = quality.compute("2025-09-16", events, doc, humans)
    os.makedirs(os.path.join(R, "eval", "quality", "fixtures"), exist_ok=True)
    with open(os.path.join(R, "eval", "quality", "fixtures", "expected.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"judge rows {len(rows)}, humans {len(humans)}, expected written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
