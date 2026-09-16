"""The judge: Claude (subscription, role=judge) scores answers ten per call against the rubric; the
execution oracle and the users' ratings are joined, never shown to it; a non-Claude signal (the Qwen3
reranker on Modal) gives a second opinion where it can.

    python eval/judge/judge.py --events <events.jsonl> [--humans eval/humans/runs/D.jsonl]
                               --out eval/judge/runs/D.json [--backend claude|none] [--modal] [--limit N]

Rows: one per navigator answer (tier nav-*), with the bank id when the question matches the bank,
the oracle verdict (a returned pointer names a gold entity; for probes, the expected terminal), the
judge's four scores, the reranker's relevance of the answer to the reference, and the dispute flag
when the two families disagree. calibrate.py turns the rows into κ.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "app"))
sys.path.insert(0, R)

import claude_cli  # noqa: E402

RUBRIC = os.path.join(HERE, "rubric.md")
BATCH = 10
_JSON_BLOCK = re.compile(r"```json\s*(\[.*?\])\s*```", re.S)
_TOK = re.compile(r"[a-z0-9]+")


# ----------------------------------------------------------------------------- bank

def load_bank():
    rows = [json.loads(l) for l in open(os.path.join(R, "eval", "q", "mfq_all.jsonl"), encoding="utf-8") if l.strip()]
    probes = [json.loads(l) for l in open(os.path.join(R, "eval", "q", "probes_offdist.jsonl"), encoding="utf-8") if l.strip()]
    return rows, probes


def _toks(s):
    return set(_TOK.findall((s or "").lower()))


def match_question(q, bank, probes, threshold=0.5):
    """-> (kind, row) by Jaccard over the question and its aliases; None when nothing is close."""
    best, best_s, kind = None, 0.0, None
    qt = _toks(q)
    if not qt:
        return None, None
    for row in bank:
        for cand in [row["q"]] + list(row.get("aliases") or []):
            ct = _toks(cand)
            s = len(qt & ct) / max(1, len(qt | ct))
            if s > best_s:
                best, best_s, kind = row, s, "bank"
    for row in probes:
        ct = _toks(row["q"])
        s = len(qt & ct) / max(1, len(qt | ct))
        if s > best_s:
            best, best_s, kind = row, s, "probe"
    return (kind, best) if best_s >= threshold else (None, None)


def gold_entities(row):
    names = set()
    for r in row.get("gold_result_excerpt") or []:
        if isinstance(r, (list, tuple)) and r:
            names.add(str(r[0]))
    for m in re.findall(r"[A-Za-z0-9_\-]+\.(?:java|ts|html|scss|yml|yaml|feature|properties)", row.get("gold_answer") or ""):
        names.add(m)
    return names


WHERE_ARCHETYPES = ("locate", "impact", "flow", "boundary", "onboarding", "onboarding_path", "cohort", "health")
TOP_POINTERS = 5


def oracle(ev, kind, row):
    """The execution oracle for pointer answers: (has_oracle, success). It measures WHERE, so it applies
    to the bank's where-archetypes (a content/overview reference cannot be checked by a pointer) and to
    the probes' expected terminal; success = a gold entity among the first TOP_POINTERS pointers."""
    if kind == "bank":
        gold = gold_entities(row)
        if row.get("gold_status") == "COVERAGE_GAP" or not gold:
            return False, None
        if row.get("archetype") not in WHERE_ARCHETYPES:
            return False, None
        ptrs = [p.get("name") for p in (ev.get("pointers") or [])[:TOP_POINTERS]]
        # WHERE, and only where: a gold entity among the first pointers. Whether the navigator then answered
        # or abstained is a different question (the `answered` flag beside it); the judge's `located` is
        # calibrated against this, its `correct` against answered-and-located.
        return True, bool(set(ptrs) & gold)
    if kind == "probe":
        return True, (ev.get("terminal") == row.get("expect"))
    return False, None


# ----------------------------------------------------------------------------- rows

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def invalidated_ids():
    """Bank rows the pack or the curated invalidation marks stale: their gold predates the graph they would judge."""
    out = set()
    for p in (os.path.join(R, "graph", "pack", "INVALIDATED_delta.json"), os.path.join(R, "eval", "q", "INVALIDATED_2026-09-02.json")):
        if os.path.exists(p):
            try:
                j = json.load(open(p, encoding="utf-8"))
                out |= {r["id"] if isinstance(r, dict) else r for r in (j if isinstance(j, list) else j.get("invalidated", []))}
            except (OSError, ValueError):
                pass
    return out


def rows_from_events(events, bank, probes, humans=None, limit=None):
    stale = invalidated_ids()
    ratings = {}
    for ev in events:
        if ev.get("event_type") == "feedback" and ev.get("rating") is not None:
            ratings[ev["request_id"]] = {"rating": ev["rating"], "verified": ev.get("verified"), "user": ev.get("user"),
                                         "tags": ev.get("tags")}
    # the personas rephrase their seed question in their own words, so text matching misses it; the night
    # file records the seed id per conversation and the request ids per turn — the first turn is the seed
    by_id = {r["id"]: ("bank", r) for r in bank}
    by_id.update({p["id"]: ("probe", p) for p in probes})
    seed_of = {}
    for h in humans or []:
        if h.get("mode") != "codemap" or not h.get("seed_id"):
            continue
        for conv in ((h.get("report") or {}).get("conversations") or []):
            turns = conv.get("turns") or []
            if turns and turns[0].get("request_id"):
                seed_of[turns[0]["request_id"]] = h["seed_id"]
    out = []
    for ev in events:
        if ev.get("event_type") != "ask" or not str(ev.get("tier", "")).startswith("nav-"):
            continue
        if ev.get("terminal") not in ("answer", "abstain"):
            continue
        kind, row = match_question(ev.get("q"), bank, probes)
        if not kind and ev.get("request_id") in seed_of and seed_of[ev["request_id"]] in by_id:
            kind, row = by_id[seed_of[ev["request_id"]]]
        has, success = oracle(ev, kind, row) if kind else (False, None)
        if kind == "bank" and row.get("id") in stale:
            has, success = False, None  # invalidated: the gold predates the pack
        answered = ev.get("terminal") == "answer"
        out.append({"id": ev["request_id"], "request_id": ev["request_id"], "user": ev.get("user"), "tier": ev.get("tier"),
                    "model": ev.get("model"), "prompt_version": ev.get("prompt_version"), "pack_version": ev.get("pack_version"),
                    "q": ev.get("q"), "answer": ev.get("answer"), "pointers": [p.get("name") for p in ev.get("pointers") or []],
                    "terminal": ev.get("terminal"), "qid": row["id"] if row else None, "kind": kind,
                    "reference": (row.get("gold_answer") if kind == "bank" else None),
                    "expect": (row.get("expect") if kind == "probe" else "answer"),
                    "oracle": {"has": has, "success": success, "answered": answered,
                               "answered_and_located": (bool(success) and answered) if has else None},
                    "human": ratings.get(ev["request_id"]), "credits": ev.get("credits"), "steps": ev.get("steps")})
    if limit:
        out = out[:limit]
    return out


# ----------------------------------------------------------------------------- judge

def batch_prompt(items):
    parts = []
    for it in items:
        parts.append(json.dumps({"id": it["id"], "question": it["q"], "terminal": it["terminal"],
                                 "answer": (it["answer"] or "")[:2500], "pointers": it["pointers"][:8],
                                 "reference": it["reference"]}, ensure_ascii=False))
    return "Score these items.\n\n" + "\n\n".join(parts)


def parse_scores(text):
    m = None
    for m in _JSON_BLOCK.finditer(text or ""):
        pass
    if not m:
        return None
    try:
        arr = json.loads(m.group(1))
    except ValueError:
        return None
    out = {}
    for o in arr:
        if not isinstance(o, dict) or "id" not in o:
            continue
        try:
            out[str(o["id"])] = {k: int(o[k]) for k in ("located", "grounded", "correct", "abstain", "helpful")}
            out[str(o["id"])]["rationale"] = str(o.get("rationale", ""))[:300]
        except (KeyError, ValueError, TypeError):
            continue
    return out


def judge_rows(rows, backend="claude", model="claude-sonnet-5", runner=None, log=print):
    usage = {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_creation": 0, "retries": 0, "errors": 0}
    if backend == "none":
        return usage
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        scores = None
        for attempt in (1, 2):
            res = claude_cli.run(batch_prompt(batch), model, role="judge", system_file=RUBRIC, max_turns=1,
                                 timeout=300, runner=runner)
            usage["calls"] += 1
            u = res.get("usage") or {}
            usage["input"] += u.get("input_tokens", 0)
            usage["output"] += u.get("output_tokens", 0)
            usage["cache_read"] += u.get("cache_read_input_tokens", 0)
            usage["cache_creation"] += u.get("cache_creation_input_tokens", 0)
            if res.get("is_error"):
                usage["errors"] += 1
                log(f"judge batch {i//BATCH}: error {res.get('error')}")
                if res.get("rate_limited"):
                    return usage
                break
            scores = parse_scores(res.get("text"))
            if scores and all(r["id"] in scores for r in batch):
                break
            usage["retries"] += 1
        for r in batch:
            r["judge"] = (scores or {}).get(r["id"])
            r["judge_backend"] = f"{backend}:{model}"
    return usage


# ----------------------------------------------------------------------------- second family

def qwen_signal(rows, client=None):
    """rr_equiv: the reranker's relevance of the answer to the reference (bank rows only)."""
    from remote import search
    client = client or search.ModalClient()
    todo = [r for r in rows if r.get("reference") and r.get("answer")]
    if not todo:
        return 0
    scores = client.rerank("Which answer states the same location and explanation as this reference? " + "REFERENCE: (per item)",
                           []) if False else None
    n = 0
    for r in todo:
        try:
            s = client.rerank(r["reference"], [r["answer"][:2000]])
            r["rr_equiv"] = float(s[0])
            n += 1
        except Exception as ex:  # a missing second opinion is recorded, not fatal
            r["rr_equiv"] = None
            r["rr_error"] = f"{type(ex).__name__}"
    return n


def flag_disputes(rows):
    """A dispute is the judge's `located` disagreeing with the execution oracle: the same question
    ("is this the right place?") answered differently by two families. The reranker's relevance is
    reported beside it, not used here: measured 2026-09-16, it sits near 1.0 whenever the topic matches."""
    n = 0
    for r in rows:
        j = r.get("judge") or {}
        jl = j.get("located")
        o = r.get("oracle") or {}
        dispute = bool(jl is not None and o.get("has") and ((jl >= 4) != bool(o.get("success"))))
        r["disputed"] = dispute
        n += dispute
    return n


def summarize(rows, usage):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None
    judged = [r for r in rows if r.get("judge")]
    return {"n": len(rows), "judged": len(judged),
            "with_oracle": sum(1 for r in rows if r["oracle"]["has"]),
            "with_human": sum(1 for r in rows if r.get("human")),
            "with_rr": sum(1 for r in rows if r.get("rr_equiv") is not None),
            "disputed": sum(1 for r in rows if r.get("disputed")),
            "mean": {k: mean([r["judge"][k] for r in judged]) for k in ("located", "grounded", "correct", "abstain", "helpful")},
            "oracle_success_rate": mean([1.0 if r["oracle"]["success"] else 0.0 for r in rows if r["oracle"]["has"]]),
            "abstain_rate": mean([1.0 if r["terminal"] == "abstain" else 0.0 for r in rows]),
            "usage": usage}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--humans", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="claude", choices=["claude", "none"])
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--modal", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    events = [json.loads(l) for l in open(a.events, encoding="utf-8") if l.strip()]
    bank, probes = load_bank()
    humans = load_jsonl(a.humans) if a.humans and os.path.exists(a.humans) else None
    rows = rows_from_events(events, bank, probes, humans=humans, limit=a.limit)
    usage = judge_rows(rows, a.backend, a.model)
    if a.modal:
        qwen_signal(rows)
    flag_disputes(rows)
    doc = {"schema": 1, "source": os.path.basename(a.events), "rows": rows, "summary": summarize(rows, usage)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True, ensure_ascii=False)
    print(json.dumps(doc["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
