"""GEPA over Erdős's manual: prompt evaluation and automatic augmentation on the five problems.

    PYTHONUTF8=1 python eval/erdos/erdos_gepa.py --label <label> [--max-metric-calls 50] [--minibatch 3] [--parallel 3]
        [--model claude-opus-5] [--judge-model claude-opus-5] [--reflection-model claude-opus-5]
        [--seed-runs 2026-09-17-v2.1] [--no-judge-repeat] [--dry-run]

The mission (owner, 2026-09-17 16:09): check whether the prompt does what it says, and augment it automatically toward
the judge's criteria. Cost and the comparison with a plain agent are not part of it.

One evaluation of a candidate on one problem: Erdős answers with the candidate manual (the pairs harness: the same
flags, the CLAUDE.md channel, the graph tool), then two instruments read the run.
- Adherence (`erdos_adherence.py`, no model): does the transcript follow the manual's rules (graph pass before files,
  key files read, facts grounded in files opened, one pass, the answer contract).
- The judge (`erdos_judge.ask_pointwise`, rubric r3, one answer at a time): correctness, completeness, architectural
  fit, and the use of the graph to understand the architecture, against the answer key, with the query trace.

    score = 0.30 correctness + 0.15 completeness + 0.20 architecture_fit + 0.20 graph_use + 0.15 adherence
            (each 1-5 score enters as (s - 1) / 4)

The five problems are the training and the validation set: in-sample by design. Three guards keep the optimiser on
behaviour: the reflector's prompt (`REFLECTION_TEMPLATE`, replacing GEPA's default, which asks the reflector to copy
every domain fact into the instruction) asks for changes in how Erdős works, within the manual's structure and size; a
candidate that names code from the answer keys, drops an include or outgrows the size limit is refused before it
runs; and the reflector reads the judge's reasons with code names masked and the answers withheld. Before optimising,
the seed's answers are judged a second time, so a gain can be read against the judge's own noise. `--resume`
continues a stopped run from GEPA's saved state.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(R, "app"))
sys.path.insert(0, os.path.join(R, "eval", "optimize"))

import erdos_adherence  # noqa: E402
import erdos_judge  # noqa: E402
import erdos_phases  # noqa: E402
import erdos_prompt  # noqa: E402
import run_pairs  # noqa: E402
import take_gold  # noqa: E402

COMPONENT = "erdos_skill"
MISSION = "prompt evaluation and automatic augmentation against the judge's criteria (owner, 2026-09-17 16:09)"
WEIGHTS = {"correctness": 0.30, "completeness": 0.15, "architecture_fit": 0.20, "graph_use": 0.20, "adherence": 0.15}
REQUIRED_INCLUDES = ("references/tools.md", "references/topology.md", "references/graph-map.md")
MAX_CHARS = 32000
TARGET_CHARS = 28000
GENERIC_STEMS = {"spec", "component", "service", "client", "routes", "interceptor", "module", "guard", "pipe", "directive",
                 "model", "models", "state", "store", "config", "types", "index", "main", "app", "test", "utils"}
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
    generic = {x for x in out if "." in x and x.split(".")[0].lower() in GENERIC_STEMS}      # "spec.ts" is a suffix, not code
    return sorted(x for x in out if len(x) >= 6 and x not in generic)


def check(body, identifiers):
    """-> problems (empty = the candidate may run)."""
    problems = []
    if not body or not body.strip():
        problems.append("empty skill body")
    if len(body or "") > MAX_CHARS:
        problems.append(f"skill body longer than {MAX_CHARS} characters ({len(body)})")
    for ref in REQUIRED_INCLUDES:
        if f'<include file="{ref}"/>' not in (body or ""):
            problems.append(f"the skill no longer includes {ref}")
    leaked = sorted({i for i in identifiers if re.search(r"(?<![A-Za-z0-9_])" + re.escape(i) + r"(?![A-Za-z0-9_])", body or "")})
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


REFLECTION_TEMPLATE = """You are improving the operating manual of Erdős, an AI architect who answers hard architectural problems about one
codebase. Erdős works with a dependency graph of that codebase, queried with Cypher, and with the source files. The
manual below reaches Erdős as its CLAUDE.md; each <include file="references/..."/> line is replaced by generated graph
data when the manual is rendered.

The current manual:
```
<curr_param>
```

Erdős answered problems with this manual. For each problem you see the problem and the feedback on the run: the
judge's scores (1-5) with reasons for correctness, completeness, architecture_fit (does the design build on the
mechanisms the codebase already uses) and graph_use (was the architecture understood from the graph before files were
read and the solution written); counts against an answer key; and deterministic checks of whether the run followed the
manual's rules. Code names in the feedback are masked and the answers are withheld.
```
<side_info>
```

Write an improved manual that raises these scores on problems like these. The rules for the new manual:
- Change behaviour, not knowledge. Improve how Erdős works: what to query and when, what to read, how to check claims
  inside the work, how to design on the mechanisms the codebase already has, how to label evidence. Add no facts about
  this codebase: no class, file, package, endpoint, table or configuration names. A manual that names code from the
  answer keys is refused before it runs.
- Keep the structure: the same XML sections, every <include file="references/..."/> line exactly as it is, the query
  recipes (edit one only if it is wrong), and the answer contract with its === ANSWER COMPLETE === marker.
- Keep the manual under 28,000 characters; one over 32,000 is refused. Sharpen or replace rules rather than adding
  new ones, and remove a rule the feedback shows does not help.
- Add no separate verification or review phase, and do not ask Erdős to double-check its answer: on this model such
  instructions cause over-verification. Checking belongs inside the work.
- Write plain imperative sentences, each with its reason. Use IMPORTANT at most once.

Return the complete new manual, and nothing else, inside ``` blocks."""


# ----------------------------------------------------------------------------- score

def score(verdict, adherence):
    """verdict: the judge's rubric r3 object (or None); adherence: the adherence score in [0, 1]."""
    if not erdos_judge.valid_pointwise(verdict):
        return 0.0
    total = sum(WEIGHTS[k] * (verdict[k] - 1) / 4 for k in erdos_judge.POINTWISE_SCORES)
    return round(total + WEIGHTS["adherence"] * max(0.0, min(1.0, adherence or 0.0)), 4)


def sha16(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------- adapter

class _EvaluationBatch:
    def __init__(self, outputs, scores, trajectories=None, objective_scores=None):
        self.outputs, self.scores, self.trajectories, self.objective_scores = outputs, scores, trajectories, objective_scores


class ErdosAdapter:
    propose_new_texts = None  # gepa 0.1.4 reads it: None = the default reflective proposal

    def __init__(self, problems, workspace, pack, run_dir, model="claude-opus-5", judge_model="claude-opus-5", max_turns=100,
                 timeout=3600, parallel=3, runner=None, judge=None, log_path=None, context_root=None):
        self.problems = {p["id"]: p for p in problems}
        self.keys = {p["id"]: take_gold.load(p["id"]) or {} for p in problems}
        self.identifiers = key_identifiers(problems)
        self.workspace, self.pack, self.run_dir = workspace, pack, run_dir
        self.model, self.judge_model, self.max_turns, self.timeout = model, judge_model, max_turns, timeout
        self.parallel = max(1, parallel)
        self.runner = runner or run_pairs.run_one          # (cmd, cwd, events_path, timeout) -> seconds
        self.judge = judge or erdos_judge.ask_pointwise     # (problem, answer, trace, model) -> (verdict, usage, is_error)
        self.log_path = log_path
        self.calls = 0
        self._lock = threading.Lock()
        os.makedirs(run_dir, exist_ok=True)
        self.mcp_path = os.path.join(run_dir, "graph_mcp.json")
        with open(self.mcp_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(run_pairs.graph_config(pack), f, indent=1)
        # candidate manuals become CLAUDE.md files, so they live outside the repository (run_pairs.context_dir)
        self.context_root = context_root or os.path.join(run_pairs.tempfile.gettempdir(), "codemap-erdos", "gepa")

    def _log(self, rec):
        if self.log_path:
            rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def candidate_dir(self, body):
        sha = sha16(body)
        path = os.path.join(self.run_dir, sha)
        os.makedirs(path, exist_ok=True)
        record = os.path.join(path, "skill_body.md")
        if not os.path.exists(record):
            with open(record, "w", encoding="utf-8", newline="\n") as f:
                f.write(body)
        return sha, path

    def judged(self, pid, answer, trace, verdict_path):
        """-> (verdict, error); one judge call per candidate and problem, one retry on an unusable reply."""
        if os.path.exists(verdict_path):
            cached = json.load(open(verdict_path, encoding="utf-8"))
            return cached.get("verdict"), cached.get("error")
        verdict, usage, err = None, None, True
        for _ in range(2):
            verdict, usage, err = self.judge(self.problems[pid], answer, trace, self.judge_model)
            if erdos_judge.valid_pointwise(verdict):
                err = False
                break
        if erdos_judge.valid_pointwise(verdict):
            with open(verdict_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"rubric": erdos_judge.POINTWISE_RUBRIC, "verdict": verdict, "usage": usage, "error": err}, f, indent=1)
        return verdict, err

    def _one(self, body, pid):
        sha, cand_dir = self.candidate_dir(body)
        problem = self.problems[pid]
        events_path = os.path.join(cand_dir, f"{pid}.erdos.events.jsonl")
        if not run_pairs.finished(events_path):  # a candidate already run on a problem is not run twice
            manual = erdos_prompt.assemble(body)
            context = run_pairs.context_dir(os.path.join(self.context_root, sha), manual)
            cmd = run_pairs.command("erdos", problem, self.model, self.max_turns, {"mcp": self.mcp_path, "context": context})
            self.runner(cmd, self.workspace, events_path, self.timeout)
        events = erdos_phases.load(events_path)
        answer = erdos_phases.split(events)["answer"] or ""
        adherence = erdos_adherence.check_run(events, answer, self.keys[pid])
        verdict, err = self.judged(pid, answer, erdos_adherence.trace_text(events), os.path.join(cand_dir, f"{pid}.verdict-r3.json"))
        s = score(verdict, adherence["score"])
        out = {"problem": pid, "candidate": sha, "score": s, "verdict": verdict, "judge_error": err, "adherence": adherence}
        self._log({"event": "eval", "candidate": sha, "problem": pid, "score": s,
                   "scores": {k: (verdict or {}).get(k) for k in erdos_judge.POINTWISE_SCORES},
                   "adherence": adherence["score"], "checks": {k: c["value"] for k, c in adherence["checks"].items()},
                   "judge_error": err})
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
            self._log({"event": "refused", "candidate": sha16(body), "problems": problems})
            return EvaluationBatch(outputs=outs, scores=[0.0] * len(pids),
                                   trajectories=[{"out": o, "feedback": "refused before running: " + "; ".join(problems)} for o in outs]
                                   if capture_traces else None)
        results = [None] * len(pids)
        sem = threading.Semaphore(self.parallel)

        def work(i, pid):
            with sem:
                results[i] = self._one(body, pid)

        threads = [threading.Thread(target=work, args=(i, pid)) for i, pid in enumerate(pids)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.calls += len(pids)
        return EvaluationBatch(outputs=results, scores=[r["score"] for r in results],
                               trajectories=[{"out": r, "feedback": self.feedback(r)} for r in results] if capture_traces else None)

    def feedback(self, r):
        v = r.get("verdict") or {}
        key = self.keys.get(r.get("problem")) or {}
        reasons = v.get("reasons") or {}
        lines = [f"score {r.get('score')} (weights: " + ", ".join(f"{k} {w}" for k, w in WEIGHTS.items()) + ")"]
        if not erdos_judge.valid_pointwise(v):
            lines.append("the judge could not grade this answer")
        for k in erdos_judge.POINTWISE_SCORES:
            lines.append(f"{k} {v.get(k)}/5: {mask(reasons.get(k) or '')}")
        lines.append(f"counts: key files named {v.get('must_find_hits')}/{len(key.get('must_find') or [])}, "
                     f"key facts {v.get('key_facts_supported')}/{len(key.get('key_facts') or [])}, "
                     f"gaps found {v.get('gaps_found')}/{len(key.get('gaps') or [])}, wrong claims {v.get('red_flags_made')}, "
                     f"existing mechanisms followed {v.get('patterns_followed')}/{len(key.get('architecture') or [])}")
        adherence = r.get("adherence") or {"checks": {}}
        lines.append(f"adherence to the manual {adherence.get('score')}: " +
                     "; ".join(f"{k} {c['value']} ({c['seen']})" for k, c in adherence["checks"].items()))
        return "\n".join(lines)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        rows = []
        for t in eval_batch.trajectories or []:
            out = t["out"]
            problem = self.problems.get(out.get("problem"), {})
            rows.append({"Inputs": mask(problem.get("prompt", "")), "Generated Outputs": "(answer withheld: graded against a key)",
                         "Feedback": t["feedback"]})
        return {c: rows for c in components_to_update}


# ----------------------------------------------------------------------------- seed runs and judge noise

def import_seed_runs(label, seed_body, run_dir, problem_ids, runs_dir=None):
    """Copy the Erdős runs of an earlier label into the seed candidate's directory when that label ran the same manual.
    -> the problem ids copied, or a reason when nothing was copied."""
    runs_dir = runs_dir or os.path.join(HERE, "runs")
    doc_path = os.path.join(runs_dir, f"{label}.json")
    if not os.path.exists(doc_path):
        return f"no run summary for {label}"
    ran = json.load(open(doc_path, encoding="utf-8")).get("meta", {}).get("prompt_version")
    wanted = erdos_prompt.version(erdos_prompt.assemble(seed_body))
    if ran != wanted:
        return f"{label} ran {ran}, the seed renders {wanted}"
    dst = os.path.join(run_dir, sha16(seed_body))
    os.makedirs(dst, exist_ok=True)
    copied = []
    for pid in problem_ids:
        src = os.path.join(runs_dir, label, f"{pid}.erdos.events.jsonl")
        target = os.path.join(dst, f"{pid}.erdos.events.jsonl")
        if run_pairs.finished(src) and not os.path.exists(target):
            shutil.copyfile(src, target)
            copied.append(pid)
    return copied


def judge_noise(ad, seed_body):
    """Grade the seed's answers a second time -> {criterion: mean absolute difference}, plus the score difference."""
    sha, cand_dir = ad.candidate_dir(seed_body)
    diffs = {k: [] for k in erdos_judge.POINTWISE_SCORES}
    score_diffs = []
    for pid in ad.problems:
        events_path = os.path.join(cand_dir, f"{pid}.erdos.events.jsonl")
        first_path = os.path.join(cand_dir, f"{pid}.verdict-r3.json")
        if not (run_pairs.finished(events_path) and os.path.exists(first_path)):
            continue
        events = erdos_phases.load(events_path)
        answer = erdos_phases.split(events)["answer"] or ""
        adherence = erdos_adherence.check_run(events, answer, ad.keys[pid])
        first = json.load(open(first_path, encoding="utf-8"))["verdict"]
        second, _ = ad.judged(pid, answer, erdos_adherence.trace_text(events), os.path.join(cand_dir, f"{pid}.verdict-r3-repeat.json"))
        if not erdos_judge.valid_pointwise(second):
            continue
        for k in diffs:
            diffs[k].append(abs(first[k] - second[k]))
        score_diffs.append(abs(score(first, adherence["score"]) - score(second, adherence["score"])))
    out = {k: round(sum(v) / len(v), 2) for k, v in diffs.items() if v}
    out["score"] = round(sum(score_diffs) / len(score_diffs), 4) if score_diffs else None
    out["problems"] = len(score_diffs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--workspace", default="C:/Users/Norbert/erdos-ws")
    ap.add_argument("--pack", default=os.path.join(R, "graph", "pack"))
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--judge-model", default="claude-opus-5")
    ap.add_argument("--reflection-model", default="claude-opus-5")
    ap.add_argument("--max-metric-calls", type=int, default=50)
    ap.add_argument("--minibatch", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-runs", default="2026-09-17-v2.1", help="reuse this label's Erdős runs for the seed when it ran the same manual")
    ap.add_argument("--no-judge-repeat", action="store_true", help="skip grading the seed's answers twice")
    ap.add_argument("--dry-run", action="store_true", help="check the seed and the seed runs; no model call")
    ap.add_argument("--resume", action="store_true", help="continue the run saved under this label")
    a = ap.parse_args(argv)
    problems = run_pairs.load_problems()
    run_dir = os.path.join(HERE, "runs", a.label)
    if os.path.exists(os.path.join(run_dir, "gepa")) and not a.resume:
        raise SystemExit(f"{run_dir}/gepa exists: GEPA would resume that run silently; pass --resume or pick a new label")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "gepa.log.jsonl")
    seed_body = erdos_prompt.skill_body()
    ad = ErdosAdapter(problems, a.workspace, a.pack, run_dir, model=a.model, judge_model=a.judge_model, max_turns=a.max_turns,
                      parallel=a.parallel, log_path=log_path)
    seed_problems = check(seed_body, ad.identifiers)
    print(f"identifiers guarded: {len(ad.identifiers)}; seed {sha16(seed_body)} check: {seed_problems or 'ok'}")
    if seed_problems:
        raise SystemExit("the seed skill violates the guard")
    seeded = import_seed_runs(a.seed_runs, seed_body, run_dir, [p["id"] for p in problems]) if a.seed_runs else "not asked"
    print("seed runs imported:", seeded)
    if a.dry_run:
        return 0
    data = [{"id": p["id"]} for p in problems]
    t0 = time.time()
    seed_eval = ad.evaluate(data, {COMPONENT: seed_body})
    print("seed scores:", dict(zip([d["id"] for d in data], seed_eval.scores)), flush=True)
    noise = None if a.no_judge_repeat else judge_noise(ad, seed_body)
    print("judge noise on the seed (mean absolute difference between two gradings):", noise, flush=True)

    import gepa
    import adapter as nav_adapter  # eval/optimize: the tool-less claude -p reflection LM
    teacher = nav_adapter.reflection_lm(a.reflection_model, log_path=log_path, timeout=900)
    result = gepa.optimize(seed_candidate={COMPONENT: seed_body}, trainset=data, valset=data, adapter=ad, reflection_lm=teacher,
                           max_metric_calls=a.max_metric_calls, reflection_minibatch_size=a.minibatch, seed=a.seed,
                           reflection_prompt_template={COMPONENT: REFLECTION_TEMPLATE},
                           run_dir=os.path.join(run_dir, "gepa"), display_progress_bar=False, raise_on_exception=False,
                           track_best_outputs=False, skip_perfect_score=False)
    scores = [round(float(s), 4) for s in (result.val_aggregate_scores or [])]
    bodies = [c[COMPONENT] if isinstance(c, dict) else str(c) for c in result.candidates]
    best_idx = int(result.best_idx) if result.best_idx is not None else 0
    with open(os.path.join(run_dir, "best_skill_body.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(bodies[best_idx])
    doc = {"schema": 2, "label": a.label, "mission": MISSION, "weights": WEIGHTS, "model": a.model, "judge_model": a.judge_model,
           "judge_rubric": erdos_judge.POINTWISE_RUBRIC, "reflection_model": a.reflection_model, "seed_runs": seeded,
           "judge_noise": noise, "candidates": [{"index": i, "sha": sha16(b), "parents": (result.parents or [None] * len(bodies))[i],
                                                 "val_score": scores[i] if i < len(scores) else None, "chars": len(b)}
                                                for i, b in enumerate(bodies)],
           "seed_val_score": scores[0] if scores else None, "best_idx": best_idx,
           "best_val_score": scores[best_idx] if scores else None, "best_check": check(bodies[best_idx], ad.identifiers),
           "metric_calls": ad.calls, "reflections": teacher.usage, "seconds": round(time.time() - t0, 1), "in_sample": True,
           "improved": bool(best_idx != 0 and scores and scores[best_idx] > scores[0])}
    with open(os.path.join(HERE, "runs", f"{a.label}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True, default=str)
    print(json.dumps({k: doc[k] for k in ("seed_val_score", "best_idx", "best_val_score", "metric_calls", "seconds", "improved")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
