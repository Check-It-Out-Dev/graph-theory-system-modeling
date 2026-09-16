"""One GEPA iteration over the navigator template, on request.

    python eval/optimize/run.py --date 2026-09-16 --train 8 --val 8 --max-metric-calls 60 [--pack graph/pack]
                                [--model claude-sonnet-5] [--reflection-model claude-sonnet-5] [--seed 0]
                                [--out eval/optimize/runs/<date>.json] [--candidate-out eval/optimize/runs/<date>.template.md]

Every model call is `claude -p` on the subscription (navigator runs as role optimizer, the teacher as
role reflector); the score is the execution oracle, never a model's opinion of itself. The artifact is
one JSON per run: seed and best validation scores, the metric calls spent, the tokens, the candidate's
sha16 and constraint status, and the lineage — enough for `promote.py` to decide and for the record to
say "no win" honestly. Runs on the box or inside `modal_gepa.py` (the same file, the same flags).
"""

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import adapter as adapter_mod  # noqa: E402
import constraints  # noqa: E402


def sha16(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d", time.gmtime()))
    ap.add_argument("--pack", default=os.path.join(R, "graph", "pack"))
    ap.add_argument("--template", default=os.path.join(R, "prompts", "navigator", "template.md"))
    ap.add_argument("--notes", default=os.path.join(R, "prompts", "navigator", "curation_notes.md"))
    ap.add_argument("--probes", default=os.path.join(R, "eval", "q", "probes_offdist.jsonl"))
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--val", type=int, default=8)
    ap.add_argument("--max-metric-calls", type=int, default=60)
    ap.add_argument("--minibatch", type=int, default=3)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--reflection-model", default="claude-sonnet-5")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--candidate-out", default=None)
    ap.add_argument("--run-dir", default=None, help="GEPA checkpoints (resume by pointing at the same dir)")
    ap.add_argument("--dry-run", action="store_true", help="evaluate the seed on the val set only; no optimisation")
    a = ap.parse_args(argv)

    out_path = a.out or os.path.join(HERE, "runs", f"{a.date}.json")
    cand_path = a.candidate_out or os.path.join(HERE, "runs", f"{a.date}.template.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_path = os.path.join(os.path.dirname(out_path), f"{a.date}.log.jsonl")  # beside the artifact, wherever it lives
    template = open(a.template, encoding="utf-8").read().replace("\r\n", "\n")
    seed_problems = constraints.check(template)
    if seed_problems:
        raise SystemExit("the seed template violates its own constraints: " + "; ".join(seed_problems))

    inval = set()
    inv_p = os.path.join(a.pack, "INVALIDATED_delta.json")
    if os.path.exists(inv_p):
        inval = set(json.load(open(inv_p, encoding="utf-8")).get("invalidated", []))
    train, val = adapter_mod.dataset(a.pack, a.probes, a.train, a.val, seed=a.seed, invalidated=inval)
    ad = adapter_mod.NavigatorAdapter(a.pack, a.notes, model=a.model, max_turns=a.max_turns, effort=a.effort, log_path=log_path)
    t0 = time.time()
    seed_cand = {adapter_mod.COMPONENT: template}
    doc = {"schema": 1, "date": a.date, "seed_template_sha16": sha16(template), "model": a.model, "reflection_model": a.reflection_model,
           "train": [i["id"] for i in train], "val": [i["id"] for i in val], "max_metric_calls": a.max_metric_calls, "seed": a.seed}

    if a.dry_run:
        eb = ad.evaluate(val, seed_cand)
        doc.update({"mode": "dry-run", "seed_val_score": round(sum(eb.scores) / max(1, len(eb.scores)), 4),
                    "per_example": [{"id": i["id"], "score": s, "terminal": o["terminal"]} for i, s, o in zip(val, eb.scores, eb.outputs)],
                    "metric_calls": ad.calls, "usage": ad.usage, "seconds": round(time.time() - t0, 1), "win": False})
        json.dump(doc, open(out_path, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
        print(json.dumps({k: doc[k] for k in ("mode", "seed_val_score", "metric_calls", "seconds")}))
        return 0

    import gepa
    teacher = adapter_mod.reflection_lm(a.reflection_model, log_path=log_path)
    result = gepa.optimize(seed_candidate=seed_cand, trainset=train, valset=val, adapter=ad, reflection_lm=teacher,
                           max_metric_calls=a.max_metric_calls, reflection_minibatch_size=a.minibatch, seed=a.seed,
                           run_dir=a.run_dir, display_progress_bar=False, raise_on_exception=False,
                           track_best_outputs=False, skip_perfect_score=True)
    scores = list(result.val_aggregate_scores or [])
    best_idx = int(result.best_idx) if result.best_idx is not None else 0
    best_text = result.best_candidate[adapter_mod.COMPONENT] if isinstance(result.best_candidate, dict) else str(result.best_candidate)
    best_problems = constraints.check(best_text)
    seed_score = scores[0] if scores else None
    best_score = scores[best_idx] if scores else None
    doc.update({"mode": "gepa", "candidates": len(result.candidates), "parents": result.parents,
                "val_scores": [round(s, 4) for s in scores], "seed_val_score": seed_score, "best_idx": best_idx,
                "best_val_score": best_score, "best_template_sha16": sha16(best_text), "best_constraints": best_problems,
                "metric_calls": int(result.total_metric_calls or ad.calls), "usage": ad.usage, "reflections": teacher.usage,
                "seconds": round(time.time() - t0, 1),
                "win": bool(best_idx != 0 and best_score is not None and seed_score is not None and best_score > seed_score and not best_problems)})
    with open(cand_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(best_text)
    json.dump(doc, open(out_path, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print(json.dumps({k: doc[k] for k in ("mode", "candidates", "seed_val_score", "best_val_score", "best_idx", "metric_calls", "seconds", "win")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
