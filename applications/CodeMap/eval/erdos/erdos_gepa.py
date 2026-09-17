"""GEPA over the Erdős skill: the skill body changes, the graph map and the tool contract stay data.

    PYTHONUTF8=1 python eval/erdos/erdos_gepa.py --label 2026-09-17-gepa [--reference 2026-09-17] [--max-metric-calls 25]
                                                 [--minibatch 3] [--model claude-opus-5] [--reflection-model claude-opus-5]
                                                 [--parallel 2] [--dry-run]

One evaluation of a candidate on one problem = Erdős answers it with the candidate skill (the pairs
harness, same flags as the pairs), then a blind judge compares that answer with the general agent's
reference answer from the reference label, against the answer key. The score puts quality first:

    score = 0.7 * min(1, erdos_overall / general_overall)     parity of judged quality
          + 0.1 * [erdos_overall >= general_overall]           reaching it
          + 0.2 * clip(1 - erdos_weighted / general_weighted)  the saving in price-weighted solve tokens

so a candidate that matches the general agent's quality at the seed's cost (about half) scores about
0.9, and a cheap answer that is judged worse cannot outscore an equally cheap one that is not.

Five problems are the whole set, so training and validation are the same problems and a win here is
in-sample. Two guards keep the optimiser honest about that: a candidate that contains a code identifier
or file name from the answer keys is refused before it runs (it would be memorising answers, not
changing behaviour), and the feedback the reflector reads has code identifiers masked, so what can
transfer is process (how much to read, when to verify, when to stop), not facts about this codebase.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(R, "app"))
sys.path.insert(0, os.path.join(R, "eval", "optimize"))

import erdos_judge  # noqa: E402
import erdos_phases  # noqa: E402
import erdos_prompt  # noqa: E402
import run_pairs  # noqa: E402
import take_gold  # noqa: E402

COMPONENT = "erdos_skill"
REQUIRED_INCLUDES = ("references/tools.md", "references/topology.md", "references/graph-map.md")
MAX_CHARS = 24000
FILE_RX = re.compile(r"[A-Za-z0-9_\-]+\.(?:java|ts|html|scss|yml|yaml|xml|properties|feature|sql|js|json|md)\b")
CAMEL_RX = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+){1,}\b")
SNAKE_RX = re.compile(r"\b[a-z]+(?:_[a-z0-9]+){1,}\b")
LOWER_CAMEL_RX = re.compile(r"\b[a-z][a-z0-9]+(?:[A-Z][a-z0-9]+){1,}\b")


# ----------------------------------------------------------------------------- guards

def key_identifiers(problems):
    """Code identifiers the answer keys name: file names and their stems, CamelCase class and method-like names."""
    out = set()
    for p in problems:
        key = take_gold.load(p["id"]) or {}
        text = json.dumps(key, ensure_ascii=False)
        for m in key.get("must_find") or []:
            out.add(m["name"])
            out.add(os.path.splitext(m["name"])[0])
        out.update(FILE_RX.findall(text))
        out.update(w for w in CAMEL_RX.findall(text) if len(w) >= 8)
        out.update(w for w in LOWER_CAMEL_RX.findall(text) if len(w) >= 8)
    return sorted(x for x in out if len(x) >= 6)


def check(body, identifiers):
    """-> problems (empty = the candidate may run)."""
    problems = []
    if not body or not body.strip():
        problems.append("empty skill body")
    if len(body) > MAX_CHARS:
        problems.append(f"skill body longer than {MAX_CHARS} characters ({len(body)})")
    for ref in REQUIRED_INCLUDES:
        if f'<include file="{ref}"/>' not in body:
            problems.append(f"the skill no longer includes {ref}")
    leaked = sorted({i for i in identifiers if re.search(r"(?<![A-Za-z0-9_])" + re.escape(i) + r"(?![A-Za-z0-9_])", body)})
    if leaked:
        problems.append("the skill names code from the answer keys (memorising, not behaviour): " + ", ".join(leaked[:8]))
    return problems


def mask(text):
    """Code identifiers out, process words in: what the reflector may learn from."""
    if not text:
        return text
    text = FILE_RX.sub("<file>", text)
    text = CAMEL_RX.sub("<code>", text)
    text = LOWER_CAMEL_RX.sub("<code>", text)
    text = SNAKE_RX.sub("<code>", text)
    return text


# ----------------------------------------------------------------------------- score

def score(erdos, general, erdos_weighted, general_weighted):
    """erdos/general: the judge's score dicts for the two answers; weighted: price-weighted solve tokens."""
    eo, go = (erdos or {}).get("overall"), (general or {}).get("overall")
    if not eo or not go:
        return 0.0
    parity = min(1.0, eo / go)
    reached = 1.0 if eo >= go else 0.0
    saving = max(0.0, min(1.0, 1.0 - erdos_weighted / general_weighted)) if general_weighted else 0.0
    return round(0.7 * parity + 0.1 * reached + 0.2 * saving, 4)


# ----------------------------------------------------------------------------- adapter

class _EvaluationBatch:
    def __init__(self, outputs, scores, trajectories=None, objective_scores=None):
        self.outputs, self.scores, self.trajectories, self.objective_scores = outputs, scores, trajectories, objective_scores


class ErdosAdapter:
    propose_new_texts = None  # gepa 0.1.4 reads it: None = the default reflective proposal

    def __init__(self, problems, reference_label, workspace, pack, run_dir, model="claude-opus-5", judge_model="claude-opus-5",
                 max_turns=100, timeout=3600, parallel=2, runner=None, judge=None, log_path=None, context_root=None):
        self.problems = {p["id"]: p for p in problems}
        self.identifiers = key_identifiers(problems)
        ref = json.load(open(os.path.join(HERE, "runs", f"{reference_label}.json"), encoding="utf-8"))
        self.reference = {r["problem"]: r for r in ref["rows"] if r["arm"] == "general"}
        self.workspace, self.pack, self.run_dir = workspace, pack, run_dir
        self.model, self.judge_model, self.max_turns, self.timeout = model, judge_model, max_turns, timeout
        self.parallel = max(1, parallel)
        self.runner = runner or run_pairs.run_one          # (cmd, cwd, events_path, timeout) -> seconds
        self.judge = judge or erdos_judge.ask               # (problem, answer_a, answer_b, model) -> (verdict, usage, error)
        self.log_path = log_path
        self.calls = 0
        os.makedirs(run_dir, exist_ok=True)
        self.mcp_path = os.path.join(run_dir, "graph_mcp.json")
        with open(self.mcp_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(run_pairs.graph_config(pack), f, indent=1)
        # candidate manuals become CLAUDE.md files, so they live outside the repository (run_pairs.context_dir)
        self.context_root = context_root or os.path.join(run_pairs.tempfile.gettempdir(), "codemap-erdos", "gepa")

    def _log(self, rec):
        if self.log_path:
            rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _one(self, body, sha, pid):
        problem = self.problems[pid]
        cand_dir = os.path.join(self.run_dir, sha)
        os.makedirs(cand_dir, exist_ok=True)
        manual = erdos_prompt.assemble(body)
        record = os.path.join(cand_dir, "erdos-manual.md")          # what ran, under a name no harness loads
        if not os.path.exists(record):
            with open(record, "w", encoding="utf-8", newline="\n") as f:
                f.write(manual)
        context = run_pairs.context_dir(os.path.join(self.context_root, sha), manual)
        events = os.path.join(cand_dir, f"{pid}.erdos.events.jsonl")
        if not run_pairs.finished(events):  # a candidate already evaluated on a problem is not run twice
            cmd = run_pairs.command("erdos", problem, self.model, self.max_turns, {"mcp": self.mcp_path, "context": context})
            self.runner(cmd, self.workspace, events, self.timeout)
        split = erdos_phases.split(erdos_phases.load(events))
        answer = split["answer"] + ("\n\n## Corrections after verification\n\n" + split["corrections"] if split["corrections"] else "")
        ref_row = self.reference[pid]
        ref_answer = open(os.path.join(R, ref_row["answer_file"]), encoding="utf-8").read()
        order = erdos_judge.blind_order(pid)
        a, b = (ref_answer, answer) if order[0] == "general" else (answer, ref_answer)
        verdict_path = os.path.join(cand_dir, f"{pid}.verdict.json")
        if os.path.exists(verdict_path):  # one verdict per candidate and problem: re-evaluations read it back
            cached = json.load(open(verdict_path, encoding="utf-8"))
            verdict, usage, err = cached.get("verdict"), cached.get("usage"), cached.get("error")
        else:
            verdict, usage, err = self.judge(problem, a, b, self.judge_model)
            if verdict:
                with open(verdict_path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump({"verdict": verdict, "usage": usage, "error": err, "blind_order": list(order)}, f, indent=1)
        letters = {order[0]: "A", order[1]: "B"}
        e_scores = (verdict or {}).get(letters["erdos"]) or {}
        g_scores = (verdict or {}).get(letters["general"]) or {}
        s = score(e_scores, g_scores, split["solve"]["tokens_weighted"], ref_row["solve"]["tokens_weighted"])
        out = {"problem": pid, "score": s, "erdos": e_scores, "general": g_scores, "why": (verdict or {}).get("why"),
               "judge_error": err, "solve": split["solve"], "verify": split["verify"], "marker": split["marker"],
               "reference_solve_weighted": ref_row["solve"]["tokens_weighted"], "is_error": split["result"].get("is_error")}
        self._log({"event": "eval", "candidate": sha, "problem": pid, "score": s, "erdos_overall": e_scores.get("overall"),
                   "general_overall": g_scores.get("overall"), "solve_weighted": split["solve"]["tokens_weighted"],
                   "file_tools": split["solve"]["tool_calls"] - split["solve"]["graph_tool_calls"],
                   "graph_tools": split["solve"]["graph_tool_calls"], "judge_error": err})
        return out

    def evaluate(self, batch, candidate, capture_traces=False):
        try:
            from gepa.core.adapter import EvaluationBatch
        except ImportError:
            EvaluationBatch = _EvaluationBatch
        body = candidate[COMPONENT]
        problems = check(body, self.identifiers)
        pids = [inst["id"] if isinstance(inst, dict) else inst for inst in batch]
        if problems:
            outs = [{"problem": pid, "score": 0.0, "refused": problems} for pid in pids]
            self._log({"event": "refused", "problems": problems})
            return EvaluationBatch(outputs=outs, scores=[0.0] * len(pids),
                                   trajectories=[{"out": o, "feedback": "refused before running: " + "; ".join(problems)} for o in outs]
                                   if capture_traces else None)
        sha = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        results = [None] * len(pids)
        sem = threading.Semaphore(self.parallel)

        def work(i, pid):
            with sem:
                results[i] = self._one(body, sha, pid)

        threads = [threading.Thread(target=work, args=(i, pid)) for i, pid in enumerate(pids)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.calls += len(pids)
        return EvaluationBatch(outputs=results, scores=[r["score"] for r in results],
                               trajectories=[{"out": r, "feedback": self.feedback(r)} for r in results] if capture_traces else None)

    def feedback(self, r):
        e, g, s = r.get("erdos") or {}, r.get("general") or {}, r.get("solve") or {}
        dims = ", ".join(f"{k} {e.get(k)} vs {g.get(k)}" for k in ("overall", "correctness", "design", "plan", "key_facts_supported",
                                                                  "gaps_found", "red_flags_made", "must_find_hits"))
        cost = (f"solve phase: {s.get('calls')} calls, {s.get('tool_calls', 0) - s.get('graph_tool_calls', 0)} file-tool calls, "
                f"{s.get('graph_tool_calls')} graph-tool calls, {round((s.get('result_bytes') or 0) / 1000)} KB read, "
                f"{s.get('tokens_weighted')} weighted tokens against the reference's {r.get('reference_solve_weighted')}")
        return (f"score {r.get('score')}. Judged Erdős vs the general agent's reference answer: {dims}. {cost}. "
                f"Judge's reasons (code names masked): {mask(r.get('why') or '')}")

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        rows = []
        for t in eval_batch.trajectories or []:
            out = t["out"]
            problem = self.problems.get(out.get("problem"), {})
            rows.append({"Inputs": mask(problem.get("prompt", "")), "Generated Outputs": "(answer withheld: judged against a key)",
                         "Feedback": t["feedback"]})
        return {c: rows for c in components_to_update}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--reference", default="2026-09-17")
    ap.add_argument("--workspace", default="C:/Users/Norbert/erdos-ws")
    ap.add_argument("--pack", default=os.path.join(R, "graph", "pack"))
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--judge-model", default="claude-opus-5")
    ap.add_argument("--reflection-model", default="claude-opus-5")
    ap.add_argument("--max-metric-calls", type=int, default=25)
    ap.add_argument("--minibatch", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="refuse-check the seed and print the identifiers guard; no model call")
    a = ap.parse_args(argv)
    problems = run_pairs.load_problems()
    run_dir = os.path.join(HERE, "runs", a.label)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "gepa.log.jsonl")
    seed_body = erdos_prompt.skill_body()
    ad = ErdosAdapter(problems, a.reference, a.workspace, a.pack, run_dir, model=a.model, judge_model=a.judge_model,
                      max_turns=a.max_turns, parallel=a.parallel, log_path=log_path)
    seed_problems = check(seed_body, ad.identifiers)
    print(f"identifiers guarded: {len(ad.identifiers)}; seed check: {seed_problems or 'ok'}")
    if seed_problems:
        raise SystemExit("the seed skill violates the guard")
    if a.dry_run:
        return 0
    import gepa
    import adapter as nav_adapter  # eval/optimize: the tool-less claude -p reflection LM
    teacher = nav_adapter.reflection_lm(a.reflection_model, log_path=log_path, timeout=900)
    data = [{"id": p["id"]} for p in problems]
    t0 = time.time()
    result = gepa.optimize(seed_candidate={COMPONENT: seed_body}, trainset=data, valset=data, adapter=ad, reflection_lm=teacher,
                           max_metric_calls=a.max_metric_calls, reflection_minibatch_size=a.minibatch, seed=a.seed,
                           run_dir=os.path.join(run_dir, "gepa"), display_progress_bar=False, raise_on_exception=False,
                           track_best_outputs=False, skip_perfect_score=False)
    scores = list(result.val_aggregate_scores or [])
    best_idx = int(result.best_idx) if result.best_idx is not None else 0
    best = result.best_candidate[COMPONENT] if isinstance(result.best_candidate, dict) else str(result.best_candidate)
    with open(os.path.join(run_dir, "best_skill_body.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(best)
    doc = {"schema": 1, "label": a.label, "reference": a.reference, "model": a.model, "judge_model": a.judge_model,
           "reflection_model": a.reflection_model, "candidates": len(result.candidates), "parents": result.parents,
           "val_scores": [round(s, 4) for s in scores], "seed_val_score": scores[0] if scores else None, "best_idx": best_idx,
           "best_val_score": scores[best_idx] if scores else None, "best_check": check(best, ad.identifiers),
           "metric_calls": ad.calls, "reflections": teacher.usage, "seconds": round(time.time() - t0, 1),
           "in_sample": True, "win": bool(best_idx != 0 and scores and scores[best_idx] > scores[0])}
    with open(os.path.join(HERE, "runs", f"{a.label}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(json.dumps({k: doc[k] for k in ("candidates", "val_scores", "best_idx", "best_val_score", "metric_calls", "seconds", "win")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
