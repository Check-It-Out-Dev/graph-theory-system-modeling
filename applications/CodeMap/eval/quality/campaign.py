"""The pair campaign across nights: gains only count in pairs, and pairs accumulate.

    python eval/quality/campaign.py [--runs eval/humans/runs] [--out eval/quality/runs/campaign.json]

A gain is one persona answering one seed question twice, once without CodeMap (baseline) and once
with it, in the same night. One night yields a handful of pairs; the README's row needs thirty. This
script folds every night's pairs into one artifact: per pair the tokens ratio (baseline / CodeMap),
the turn delta and the seconds delta, both ratings, and per campaign the means, the count, and the
share of pairs where CodeMap used fewer tokens, fewer turns, or was rated at least as well. The
quality page reads it; the flip claim cites `n_pairs` from it.
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))


def _toks(r):
    u = r.get("usage") or {}
    return sum(int(u.get(k, 0) or 0) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))


def _rating(r):
    rep = r.get("report") or {}
    vals = [t.get("rating") for c in (rep.get("conversations") or []) for t in (c.get("turns") or []) if isinstance(t.get("rating"), (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else None


def pairs_of(rows, night):
    base = {(r["persona"], r["seed_id"]): r for r in rows if r.get("mode") == "baseline" and not r.get("is_error")}
    cm = {(r["persona"], r["seed_id"]): r for r in rows if r.get("mode") == "codemap" and not r.get("is_error")}
    out = []
    for k in sorted(base):
        if k not in cm:
            continue
        b, c = base[k], cm[k]
        tb, tc = _toks(b), _toks(c)
        out.append({"night": night, "persona": k[0], "seed_id": k[1], "kind": c.get("kind"),
                    "tokens_baseline": tb, "tokens_codemap": tc, "tokens_ratio": round(tb / tc, 3) if tc else None,
                    "turns_baseline": b.get("num_turns"), "turns_codemap": c.get("num_turns"),
                    "turns_delta": (b.get("num_turns") or 0) - (c.get("num_turns") or 0) if b.get("num_turns") and c.get("num_turns") else None,
                    "seconds_delta": round(((b.get("duration_ms") or 0) - (c.get("duration_ms") or 0)) / 1000.0, 1) if b.get("duration_ms") and c.get("duration_ms") else None,
                    "rating_baseline": _rating(b), "rating_codemap": _rating(c)})
    return out


def campaign(runs_dir):
    pairs = []
    nights = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*.jsonl"))):
        night = os.path.basename(p)[:-6]
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        night_pairs = pairs_of(rows, night)
        nights.append({"night": night, "conversations": len(rows), "baselines": sum(1 for r in rows if r.get("mode") == "baseline"), "pairs": len(night_pairs)})
        pairs += night_pairs

    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(sum(xs) / len(xs), 3) if xs else None

    def share(pred):
        xs = [p for p in pairs if pred(p) is not None]
        return round(sum(1 for p in xs if pred(p)) / len(xs), 3) if xs else None

    return {"schema": 1, "n_pairs": len(pairs), "nights": nights, "pairs": pairs,
            "tokens_ratio_mean": mean([p["tokens_ratio"] for p in pairs]),
            "turns_delta_mean": mean([p["turns_delta"] for p in pairs]),
            "seconds_delta_mean": mean([p["seconds_delta"] for p in pairs]),
            "rating_baseline_mean": mean([p["rating_baseline"] for p in pairs]),
            "rating_codemap_mean": mean([p["rating_codemap"] for p in pairs]),
            "share_fewer_tokens": share(lambda p: None if p["tokens_ratio"] is None else p["tokens_ratio"] > 1.0),
            "share_fewer_turns": share(lambda p: None if p["turns_delta"] is None else p["turns_delta"] > 0),
            "share_rated_at_least_as_well": share(lambda p: None if p["rating_codemap"] is None or p["rating_baseline"] is None else p["rating_codemap"] >= p["rating_baseline"]),
            "gated": len(pairs) >= 30}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(R, "eval", "humans", "runs"))
    ap.add_argument("--out", default=os.path.join(R, "eval", "quality", "runs", "campaign.json"))
    a = ap.parse_args(argv)
    doc = campaign(a.runs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(doc, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print(json.dumps({k: doc[k] for k in ("n_pairs", "tokens_ratio_mean", "turns_delta_mean", "seconds_delta_mean", "rating_baseline_mean",
                                          "rating_codemap_mean", "share_fewer_tokens", "share_fewer_turns", "share_rated_at_least_as_well", "gated")}))
    print("nights:", [(n["night"], n["pairs"]) for n in doc["nights"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
