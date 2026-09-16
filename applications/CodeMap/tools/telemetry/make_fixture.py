"""Write telemetry/fixtures/events.sample.jsonl: 200 deterministic, schema-valid, scrubbed rows.

The fixture is what CI replays (no model, no network): the metrics view, the credit ledger and
the quality rates are asserted against it. Regenerate only when the schema changes, and commit
the diff with the test that reads it.
"""

import json
import os
import random
import sys

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, R)

from remote import credits, telemetry  # noqa: E402

USERS = [("owner", "owner"), ("opus-architect", "persona"), ("opus-reviewer", "persona"), ("sonnet-bugfixer", "persona"),
         ("sonnet-newcomer", "persona"), ("haiku-pm", "persona"), ("haiku-ops", "persona"), ("anonymous", "anonymous")]
QS = ["Which class signs the consent cookies?", "Where is the Stripe webhook handled?", "What does the payments boot guard do?",
      "Which component shows the rate-limit banner?", "How are magic links validated?", "Where are invoices sent to Fakturownia?",
      "What depends on UserRepository?", "Which cron enforces consent?", "Where is the subscription state machine?"]
ENTS = ["ConsentCookieService.java", "StripeWebhookController.java", "PaymentsDisabledBootGuard.java", "RateLimitBannerComponent.ts",
        "MagicLinkService.java", "FakturowniaAdapter.java", "UserRepository.java", "ConsentEnforcementCronJob.java", "SubscriptionService.java"]
TAGS = ["great", "pointer_verified", "incomplete", "slow", "wrong", "should_have_abstained", "hallucinated", "pointer_wrong"]


def main(out=None, n=200, seed=42):
    rnd = random.Random(seed)
    card = credits.RateCard()
    out = out or os.path.join(R, "telemetry", "fixtures", "events.sample.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    asks = []
    base_ts = 1758000000  # 2025-09-16T05:20:00Z, a fixed epoch
    for i in range(n):
        ts = base_ts + i * 37
        iso = __import__("datetime").datetime.utcfromtimestamp(ts).isoformat(timespec="milliseconds") + "Z"
        user, kind = rnd.choice(USERS)
        rid = f"00000000-0000-4000-8000-{i:012d}"
        r = rnd.random()
        if r < 0.55 or not asks:
            faq = rnd.random() < 0.25
            tier = "faq" if faq else ("nav-opus" if rnd.random() < 0.15 else "nav-sonnet")
            terminal = "answer" if rnd.random() < 0.85 else "abstain"
            qi = rnd.randrange(len(QS))
            tokens = None
            credit = 0.0
            steps = 0
            if not faq:
                steps = rnd.randint(2, 7)
                tokens = {"prompt": rnd.randint(5, 40), "completion": rnd.randint(300, 1400),
                          "cached": 30000 * steps + rnd.randint(0, 5000), "cache_creation": rnd.randint(700, 30000)}
                credit = card.compute(tier, tokens)
            ev = {"schema": 1, "event_type": "ask", "ts": iso, "request_id": rid, "user": user, "user_kind": kind,
                  "tier": tier, "terminal": terminal, "q": QS[qi], "answer": f"{ENTS[qi]} in subsystem 11 (scrubbed).",
                  "pointers": [{"name": ENTS[qi], "repo": "backend", "path": f"src/main/java/x/{ENTS[qi]}", "subsystem": 11}],
                  "credits": credit, "steps": steps, "duration_ms": rnd.randint(400, 30000) if not faq else rnd.randint(5, 40),
                  "faq_cache": "hit" if faq else "miss", "protocol": "cache" if faq else "navigator",
                  "context_id": f"ctx-{i//3}", "pack_version": "1.0.0", "prompt_version": "nav@d506112a338ebf22",
                  "tool": "codemap_ask"}
            if tokens:
                ev["tokens"] = tokens
                ev["cache_read"] = tokens["cached"]
                ev["model"] = "claude-opus-5" if tier == "nav-opus" else "claude-sonnet-5"
                ev["model_version"] = ev["model"]
                ev["trajectory"] = [f"find({ENTS[qi].split('.')[0]})", f"impact({ENTS[qi]})"][:steps]
            asks.append((rid, tier, user))
        elif r < 0.80:
            arid, atier, auser = rnd.choice(asks)
            verified = rnd.random() < 0.7
            rating = rnd.choice([5, 5, 4, 4, 4, 3, 2, 1]) if verified else rnd.choice([3, 2, 1])
            ev = {"schema": 1, "event_type": "feedback", "ts": iso, "request_id": arid, "user": auser, "user_kind": kind,
                  "tier": atier, "rating": rating, "tags": rnd.sample(TAGS, rnd.randint(0, 2)), "verified": verified,
                  "comment": "scrubbed", "credits": 0.0, "duration_ms": 3, "tool": "codemap_feedback",
                  "pack_version": "1.0.0", "prompt_version": "nav@d506112a338ebf22"}
        elif r < 0.90:
            ev = {"schema": 1, "event_type": "step", "ts": iso, "request_id": rid, "user": user, "user_kind": kind,
                  "tier": "engine", "terminal": "step", "q": "map()", "credits": 0.0, "steps": 1, "duration_ms": 12,
                  "tool": "codemap_step", "pack_version": "1.0.0", "prompt_version": "nav@d506112a338ebf22"}
        elif r < 0.96:
            ev = {"schema": 1, "event_type": "miss", "ts": iso, "request_id": rid, "user": user, "user_kind": kind,
                  "tier": "none", "path": rnd.choice(["backend/src/main/resources/application.yml", "frontend/src/app/app.config.ts"]),
                  "why": "grepped for it", "credits": 0.0, "duration_ms": 2, "tool": "codemap_miss",
                  "pack_version": "1.0.0", "prompt_version": "nav@d506112a338ebf22"}
        else:
            ev = {"schema": 1, "event_type": "budget_refusal", "ts": iso, "request_id": rid, "user": user, "user_kind": kind,
                  "tier": "nav-sonnet", "terminal": "budget_exhausted", "spent": 201.5, "budget": 200.0, "credits": 0.0,
                  "duration_ms": 1, "tool": "codemap_ask", "pack_version": "1.0.0", "prompt_version": "nav@d506112a338ebf22"}
        rows.append(telemetry.validate(ev))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for ev in rows:
            f.write(json.dumps(ev, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
