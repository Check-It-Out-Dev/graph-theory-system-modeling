"""Calibration: is the judge worth believing? Cohen's κ between the judge's `located >= 4` and the
execution oracle on rows that have one (hard gate: >= 0.6), κ against the users' `rating >= 4` where
both exist (reported), agreement with the reranker signal, and anchor drift: a frozen set of items
with their first scores is re-scored every run; a mean shift above 0.3 or a κ drop above 0.1 is drift.

The κ paradox (Feinstein & Cicchetti 1990): when almost every oracle row is a hit, κ's chance term
eats the agreement — 29/34 identical verdicts can score κ 0.25. So the raw agreement, the prevalence
(mean yes-rate of the two raters) and Gwet's AC1 (a chance-corrected agreement that is stable under
skew) are reported beside κ every time, and the verdict has two routes, both printed: κ >= gate; or,
only when prevalence is outside [0.15, 0.85], agreement >= 0.8 and AC1 >= gate. `basis` names the
route that held. Neither threshold moves per night.

    python eval/judge/calibrate.py --run eval/judge/runs/D.json [--anchors eval/judge/anchors.jsonl]
                                   [--freeze-anchors N] [--gate 0.6]      exit 3 when uncalibrated, 4 on drift
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def cohen_kappa(pairs):
    """pairs: [(a, b)] of booleans -> κ (None when fewer than 2 pairs or no variance)."""
    pairs = [(bool(a), bool(b)) for a, b in pairs]
    n = len(pairs)
    if n < 2:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if abs(1.0 - pe) < 1e-12:  # no variance: agreement is all there is
        return 1.0 if abs(1.0 - po) < 1e-12 else 0.0
    return round((po - pe) / (1 - pe), 4)


def gwet_ac1(pairs):
    """Gwet's AC1 over boolean pairs: (p_o - p_e) / (1 - p_e) with p_e = 2·π·(1-π), π the mean yes-rate."""
    pairs = [(bool(a), bool(b)) for a, b in pairs]
    n = len(pairs)
    if n < 2:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pi = (sum(1 for a, _ in pairs if a) / n + sum(1 for _, b in pairs if b) / n) / 2
    pe = 2 * pi * (1 - pi)
    if abs(1.0 - pe) < 1e-12:
        return None
    return round((po - pe) / (1 - pe), 4)


def agreement_stats(pairs):
    """-> {agreement, prevalence, ac1} for the same pairs κ is computed on (None when fewer than 2)."""
    pairs = [(bool(a), bool(b)) for a, b in pairs]
    n = len(pairs)
    if n < 2:
        return {"agreement": None, "prevalence": None, "ac1": None}
    po = sum(1 for a, b in pairs if a == b) / n
    pi = (sum(1 for a, _ in pairs if a) / n + sum(1 for _, b in pairs if b) / n) / 2
    return {"agreement": round(po, 4), "prevalence": round(pi, 4), "ac1": gwet_ac1(pairs)}


def oracle_pairs(rows):
    return [(r["judge"]["located"] >= 4, r["oracle"]["success"]) for r in rows
            if r.get("judge") and r["judge"].get("located") is not None and r.get("oracle", {}).get("has")]


def kappa_oracle(rows):
    """judge.located vs the execution oracle: both answer "is this the right place?"."""
    pairs = oracle_pairs(rows)
    return cohen_kappa(pairs), len(pairs)


def verdict(kappa, stats, gate=0.6, skew=0.85):
    """-> (calibrated, basis). κ route first; the AC1 route only when prevalence is skewed past `skew`."""
    if kappa is not None and kappa >= gate:
        return True, "kappa"
    p, a, ac1 = stats.get("prevalence"), stats.get("agreement"), stats.get("ac1")
    if p is not None and (p >= skew or p <= 1 - skew) and a is not None and a >= 0.8 and ac1 is not None and ac1 >= gate:
        return True, "ac1"
    return False, None


def kappa_human(rows):
    pairs = [(r["judge"]["correct"] >= 4, r["human"]["rating"] >= 4) for r in rows
             if r.get("judge") and r.get("human") and r["human"].get("rating") is not None]
    return cohen_kappa(pairs), len(pairs)


def agreement_rr(rows):
    pairs = [(r["judge"]["correct"] >= 4, r["rr_equiv"] >= 0.5) for r in rows
             if r.get("judge") and r.get("rr_equiv") is not None]
    if not pairs:
        return None, 0
    return round(sum(1 for a, b in pairs if a == b) / len(pairs), 4), len(pairs)


def load_anchors(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def anchor_drift(rows, anchors):
    """Compare this run's scores on anchor items (by request_id) with their frozen scores."""
    by_id = {r["id"]: r for r in rows if r.get("judge")}
    deltas = []
    pairs_then, pairs_now = [], []
    for a in anchors:
        r = by_id.get(a["id"])
        if not r:
            continue
        deltas.append(abs(r["judge"]["correct"] - a["judge"]["correct"]))
        if a.get("oracle_success") is not None and "located" in a["judge"] and "located" in r["judge"]:
            pairs_then.append((a["judge"]["located"] >= 4, a["oracle_success"]))
            pairs_now.append((r["judge"]["located"] >= 4, a["oracle_success"]))
    if not deltas:
        return {"matched": 0, "mean_abs_delta": None, "kappa_then": None, "kappa_now": None, "drift": False}
    k_then, k_now = cohen_kappa(pairs_then), cohen_kappa(pairs_now)
    mad = round(sum(deltas) / len(deltas), 3)
    drift = mad > 0.3 or (k_then is not None and k_now is not None and (k_then - k_now) > 0.1)
    return {"matched": len(deltas), "mean_abs_delta": mad, "kappa_then": k_then, "kappa_now": k_now, "drift": drift}


def freeze_anchors(rows, path, n):
    picked = [r for r in rows if r.get("judge") and r.get("oracle", {}).get("has")][:n]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in picked:
            f.write(json.dumps({"id": r["id"], "qid": r.get("qid"), "q": r["q"], "judge": r["judge"],
                                "oracle_success": r["oracle"]["success"], "frozen_from": os.path.basename(path)}, ensure_ascii=False) + "\n")
    return len(picked)


def calibrate(doc, anchors, gate=0.6):
    rows = doc["rows"]
    ko, n_o = kappa_oracle(rows)
    stats = agreement_stats(oracle_pairs(rows))
    kh, n_h = kappa_human(rows)
    ar, n_r = agreement_rr(rows)
    drift = anchor_drift(rows, anchors)
    ok, basis = verdict(ko, stats, gate)
    return {"kappa_oracle": ko, "n_oracle": n_o, "agreement_oracle": stats["agreement"], "prevalence_oracle": stats["prevalence"],
            "ac1_oracle": stats["ac1"], "basis": basis, "kappa_human": kh, "n_human": n_h,
            "agreement_rr": ar, "n_rr": n_r, "anchors": drift, "gate": gate,
            "calibrated": ok,
            "verdict": ("calibrated" if ok else ("uncalibrated" if ko is not None else "no oracle rows"))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--anchors", default=os.path.join(HERE, "anchors.jsonl"))
    ap.add_argument("--freeze-anchors", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.6)
    a = ap.parse_args(argv)
    doc = json.load(open(a.run, encoding="utf-8"))
    if a.freeze_anchors:
        print(f"froze {freeze_anchors(doc['rows'], a.anchors, a.freeze_anchors)} anchors to {a.anchors}")
    result = calibrate(doc, load_anchors(a.anchors), a.gate)
    doc["calibration"] = result
    with open(a.run, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True, ensure_ascii=False)
    print(json.dumps(result, indent=1))
    if result["anchors"]["drift"]:
        return 4
    return 0 if result["calibrated"] else 3


if __name__ == "__main__":
    sys.exit(main())
