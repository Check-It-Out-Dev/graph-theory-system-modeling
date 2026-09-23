"""GEPA over a prompt under test: the conventions manual evolves on the training tasks, scored by the contract's
deterministic checks and the blind judge, until the plateau stopper (K iterations without a gain above delta) or the
metric-call budget ends the run.

    PYTHONUTF8=1 python eval/put/run.py gepa --campaign <campaign.json> --base-repo <checkout>
      campaign: {"label", "baseline": <baseline label, supplies delta and the seed's runs>, "max_metric_calls": 60,
                 "minibatch": 3, "parallel": 3, "resume": false}

One evaluation = one coding run of one training task with the candidate (the runner, the checks, the judge), cached by
(candidate sha16, task, replicate 0). The training tasks are the validation set, as with the Erdős manual: the claim
rests on the held-out certification afterwards, never on these scores (law 5). Three guards keep the optimiser on
behaviour, not memory: the reflector reads masked feedback (code names out, diffs withheld); a candidate that names an
identifier from a task's hidden tests, reference solution or interface, drops a rule id or an include, or outgrows
the size limit is refused before it runs; and the reflection prompt asks for changes in how the agent works.
"""

import json
import os
import re
import shutil
import threading
import time

import put_paths
import put_checks
import put_contract
import put_judge
import put_prompt
import put_report
import put_runner
import put_score
import put_stop
import put_tasks
import erdos_gepa            # eval/erdos: mask, FILE_RX, CAMEL_RX, LOWER_CAMEL_RX, reflection_lm, _EvaluationBatch

COMPONENT = "conventions_manual"
TARGET_CHARS = 20000

REFLECTION_TEMPLATE = """You are improving the conventions manual a team gives to its AI coding agents. The manual reaches the agent as
the CLAUDE.md of its session in a Spring Boot (Java) repository; it holds the team's rules, a section on the code graph
the agent can query, a working method and an answer contract. Each <include file="references/..."/> line is replaced by
generated reference data when the manual is rendered.

The current manual:
```
<curr_param>
```

The agent solved coding tasks with this manual. For each task you see the task and the feedback on the run: which
rules the change followed or broke (deterministic checks, with what they counted), how many of the reviewers' hidden
acceptance tests passed, and a blind reviewer's scores (1-5) with reasons for correctness, convention fit, design fit,
test quality and use of the code graph. Code names are masked and the diffs are withheld.
```
<side_info>
```

Write an improved manual that makes the agent follow the rules more reliably and write better changes on tasks like
these. The rules for the new manual:
- Change behaviour, not knowledge. Improve how the agent works: when to query the graph and what to ask it, which
  existing file to read before writing each kind of class, what to check in its own change before it finishes, how to
  test. Add no facts about this repository's tasks: no class, file, method, endpoint, table or property names from the
  tasks. A manual that names code from the tasks is refused before it runs.
- Keep every <rule id="..."> element with the same id; you may rewrite the text inside a rule, merge advice into it,
  or make it more concrete. Keep every <include file="references/..."/> line exactly as it is, and the answer contract
  with its === ANSWER COMPLETE === line.
- Keep the manual under 20,000 characters; one over 24,000 is refused. Sharpen or replace sentences rather than
  adding many new ones, and remove advice the feedback shows does not help.
- Write plain imperative sentences, each with its reason. Use IMPORTANT at most once.

Return the complete new manual, and nothing else, inside ``` blocks."""


def task_identifiers(instance, tasks):
    """Identifiers a manual may not name: the classes, files and methods of the tasks' hidden tests, reference
    solutions and interfaces (memorising a task is not a convention)."""
    out = set()
    for t in tasks:
        texts = [" ".join(t.get("interface", ()))]
        d = put_tasks.task_dir(instance, t["id"])
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith((".java", ".patch")):
                    with open(os.path.join(root, f), encoding="utf-8", errors="replace") as fh:
                        texts.append(fh.read())
        for text in texts:
            added = "\n".join(line[1:] for line in text.splitlines() if line.startswith("+")) if "diff --git" in text else text
            out.update(w for w in erdos_gepa.CAMEL_RX.findall(added) if len(w) >= 10)
            out.update(w for w in erdos_gepa.LOWER_CAMEL_RX.findall(added) if len(w) >= 12)
            out.update(re.findall(r"\b[a-z0-9-]+\.sql\b", added))
    allowed = {"RequiredArgsConstructor", "TransactionalEventListener", "SchedulerLock", "TranslatableException",
               "BusinessRuleTranslatableException", "ValidationTranslatableException", "MockitoExtension", "ExtendWith",
               "ResponseEntity", "PreAuthorize", "RateLimit", "ApplicationEventPublisher", "ConfigurationProperties",
               "TransactionPhase", "ReflectionTestUtils", "SpringBootTest", "LocalDateTime", "ConditionalOnProperty",
               "InjectMocks", "RequestBody", "PathVariable", "RequestParam", "GetMapping", "PostMapping", "PutMapping",
               "DeleteMapping", "PatchMapping", "BaseRepository", "ArgumentCaptor", "IllegalStateException",
               "IllegalArgumentException", "DisplayName", "BeforeEach", "MockitoSettings"}
    return sorted(i for i in out if i not in allowed)


def guard(body, contract, seed_body, identifiers):
    """-> problems (empty = the candidate may run)."""
    problems = []
    if not body or not body.strip():
        return ["empty manual"]
    limit = contract["prompt"]["max_body_chars"]
    if len(body) > limit:
        problems.append(f"manual longer than {limit} characters ({len(body)})")
    for inc in contract["prompt"]["required_includes"]:
        if f'<include file="{inc}"/>' not in body:
            problems.append(f"the manual no longer includes {inc}")
    missing = [r for r in put_contract.rule_ids(seed_body) if f'<rule id="{r}">' not in body]
    if missing:
        problems.append("rule ids dropped: " + ", ".join(missing))
    if contract["prompt"]["marker"] not in body:
        problems.append("the completion line is gone")
    leaked = sorted({i for i in identifiers if re.search(r"(?<![A-Za-z0-9_])" + re.escape(i) + r"(?![A-Za-z0-9_])", body)})
    if leaked:
        problems.append("the manual names code from the tasks (memorising, not behaviour): " + ", ".join(leaked[:8]))
    return problems


class PutAdapter:
    propose_new_texts = None      # gepa 0.1.4 reads it: None = the default reflective proposal

    def __init__(self, instance, contract, tasks, base_repo, label, seed_body, parallel=3, log_path=None,
                 execute=None, judge=None, repo=None):
        self.instance, self.contract, self.label = instance, contract, label
        self.tasks = {t["id"]: t for t in tasks}
        self.base_repo, self.seed_body = base_repo, seed_body
        self.identifiers = task_identifiers(instance, tasks)
        self.parallel = max(1, parallel)
        self.log_path = log_path
        self.execute = execute or put_runner.execute
        self.judge = judge or put_judge.judge_run
        self.repo = repo or put_checks.Repo(base_repo, contract["base_sha"])
        self.calls = 0
        self._lock = threading.Lock()

    def _log(self, rec):
        if self.log_path:
            rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def run_dir(self, body, task_id):
        return put_runner.run_dir_for(put_paths.RUNS, self.label, body, task_id, 0)

    def one(self, body, task_id):
        task = self.tasks[task_id]
        rd = self.run_dir(body, task_id)
        if not put_runner.finished(rd):
            self.execute(task, body, rd, self.base_repo, self.contract, self.instance, label=self.label)
        checks = put_checks.check_run(rd, task, self.contract, self.repo)
        verdict = self.judge(rd, task, self.instance, self.contract["models"]["judge"]).get("verdict")
        s, comp = put_score.score(checks, verdict, self.contract)
        with open(os.path.join(rd, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        rec = {"task": task_id, "candidate": put_prompt.sha16(body), "score": s, "components": comp,
               "checks": checks["checks"], "verdict": verdict, "tests": _read(rd, "tests.json"),
               "budget_exhausted": meta["session"].get("budget_exhausted")}
        self._log({"event": "eval", "candidate": rec["candidate"], "task": task_id, "score": s})
        return rec

    def evaluate(self, batch, candidate, capture_traces=False):
        try:
            from gepa.core.adapter import EvaluationBatch
        except ImportError:
            EvaluationBatch = erdos_gepa._EvaluationBatch
        body = candidate[COMPONENT]
        ids = [row["id"] if isinstance(row, dict) else row for row in batch]
        problems = guard(body, self.contract, self.seed_body, self.identifiers)
        if problems:
            self._log({"event": "refused", "candidate": put_prompt.sha16(body), "problems": problems})
            outs = [{"task": i, "score": 0.0, "refused": problems} for i in ids]
            return EvaluationBatch(outputs=outs, scores=[0.0] * len(ids),
                                   trajectories=[{"out": o, "feedback": "refused before running: " + "; ".join(problems)} for o in outs]
                                   if capture_traces else None)
        done, sem = {}, threading.Semaphore(self.parallel)

        def work(tid):
            with sem:
                try:
                    done[tid] = self.one(body, tid)
                except Exception as ex:                                  # one broken run must not take the batch down
                    self._log({"event": "failed", "candidate": put_prompt.sha16(body), "task": tid, "error": f"{type(ex).__name__}: {ex}"})
                    done[tid] = {"task": tid, "score": 0.0, "failed": f"{type(ex).__name__}: {ex}", "checks": {}, "verdict": None}

        threads = [threading.Thread(target=work, args=(tid,)) for tid in dict.fromkeys(ids)]    # a repeated task runs once
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        results = [done[i] for i in ids]
        self.calls += len(dict.fromkeys(ids))
        return EvaluationBatch(outputs=results, scores=[r["score"] for r in results],
                               trajectories=[{"out": r, "feedback": self.feedback(r)} for r in results] if capture_traces else None)

    def feedback(self, r):
        if r.get("failed"):
            return f"the run failed before it could be scored: {erdos_gepa.mask(r['failed'])}"
        lines = [f"score {r['score']:.3f}"]
        t = r.get("tests") or {}
        lines.append(f"build {'green' if t.get('build_green') else 'red'}; hidden acceptance tests {t.get('hidden')}; "
                     f"existing tests of the touched classes {t.get('pass_to_pass')}; the agent's own unit tests {t.get('own')}")
        if r.get("budget_exhausted"):
            lines.append("the session ran out of its output-token budget before finishing")
        for rid, c in (r.get("checks") or {}).items():
            if c.get("applicable") and c.get("value") is not None:
                lines.append(f"rule {rid}: {'followed' if c['passed'] else 'NOT followed'} ({c['value']}); {erdos_gepa.mask(c['seen'])}")
        v = r.get("verdict") or {}
        reasons = v.get("reasons") or {}
        for crit in self.contract["judge"]["criteria"]:
            lines.append(f"reviewer {crit} {v.get(crit)}/5: {erdos_gepa.mask(reasons.get(crit, ''))}")
        return "\n".join(lines)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        rows = []
        for t in eval_batch.trajectories or []:
            task = self.tasks.get(t["out"].get("task"), {})
            rows.append({"Inputs": erdos_gepa.mask(task.get("text", "")),
                         "Generated Outputs": "(diff withheld: graded by the checks and the reviewer)",
                         "Feedback": t["feedback"]})
        return {c: rows for c in components_to_update}


def _read(rd, name):
    p = os.path.join(rd, name)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def import_baseline_runs(baseline_label, label, seed_body, task_ids):
    """Reuse the baseline's replicate 0 of the seed as the GEPA run's seed evaluation: same prompt, same task."""
    sha = put_prompt.sha16(seed_body)
    copied = 0
    for tid in task_ids:
        src = os.path.join(put_paths.RUNS, baseline_label, sha, f"{tid}.r0")
        dst = os.path.join(put_paths.RUNS, label, sha, f"{tid}.r0")
        if put_runner.finished(src) and not put_runner.finished(dst):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            copied += 1
    return copied


def main_from_cli(camp, label, contract, instance, base_repo, parallel, summary_path):
    import gepa
    st = contract["statistics"]
    tasks = [t for t in put_contract.tasks(instance) if t["split"] == "train"]
    seed_body = put_prompt.load_body(instance, camp.get("seed", "v1"))
    run_dir = os.path.join(put_paths.RUNS, label)
    gepa_dir = os.path.join(run_dir, "gepa")
    if os.path.exists(gepa_dir) and not camp.get("resume"):
        raise SystemExit(f"{gepa_dir} exists: GEPA would resume it silently; set resume in the campaign or pick a new label")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "gepa.log.jsonl")
    baseline = camp.get("baseline")
    with open(os.path.join(put_paths.RUNS, f"{baseline}.json"), encoding="utf-8") as f:
        base_summary = json.load(f)
    delta = base_summary["noise"]["delta"]
    copied = import_baseline_runs(baseline, label, seed_body, [t["id"] for t in tasks])
    print(f"seed runs reused from {baseline}: {copied}; delta {delta}; K {st['plateau_k']}", flush=True)
    ad = PutAdapter(instance, contract, tasks, base_repo, label, seed_body, parallel=parallel, log_path=log_path)
    seed_problems = guard(seed_body, contract, seed_body, ad.identifiers)
    if seed_problems:
        raise SystemExit(f"the seed violates the guard: {seed_problems}")
    stopper = put_stop.PlateauStopper(st["plateau_k"], delta, os.path.join(run_dir, "plateau.json"))
    teacher = erdos_gepa.reflection_lm(contract["models"]["reflector"], timeout=2400, log_path=log_path)
    data = [{"id": t["id"]} for t in tasks]
    t0 = time.time()
    result = gepa.optimize(seed_candidate={COMPONENT: seed_body}, trainset=data, valset=data, adapter=ad, reflection_lm=teacher,
                           max_metric_calls=camp.get("max_metric_calls", st["gepa_max_metric_calls"]),
                           reflection_minibatch_size=camp.get("minibatch", st["gepa_minibatch"]), seed=0,
                           reflection_prompt_template={COMPONENT: REFLECTION_TEMPLATE}, stop_callbacks=[stopper],
                           run_dir=gepa_dir, display_progress_bar=False, raise_on_exception=False,
                           track_best_outputs=False, skip_perfect_score=False)
    scores = [round(float(s), 4) for s in (result.val_aggregate_scores or [])]
    bodies = [c[COMPONENT] if isinstance(c, dict) else str(c) for c in result.candidates]
    best = int(result.best_idx) if result.best_idx is not None else 0
    with open(os.path.join(run_dir, "best_prompt.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(bodies[best])
    reason = "plateau" if stopper.stopped else ("budget" if ad.calls >= camp.get("max_metric_calls", st["gepa_max_metric_calls"]) else "owner")
    doc = {"schema": 1, "label": label, "mode": "gepa", "instance": instance, "baseline": baseline,
           "models": contract["models"], "weights": contract["weights"], "base_sha": contract["base_sha"],
           "prompt_version": put_prompt.version(bodies[best]), "prompt_origin": "GEPA best candidate",
           "candidates": [{"index": i, "sha": put_prompt.sha16(b), "version": put_prompt.version(b),
                           "parents": (result.parents or [None] * len(bodies))[i],
                           "val_score": scores[i] if i < len(scores) else None, "chars": len(b)} for i, b in enumerate(bodies)],
           "seed_val_score": scores[0] if scores else None, "best_idx": best, "best_val_score": scores[best] if scores else None,
           "improved": bool(best != 0 and scores and scores[best] > scores[0] + delta),
           "stop": {"reason": reason, "after_iteration": stopper.last_iteration, "delta": delta, "k": st["plateau_k"],
                    "streak": stopper.streak(), "anchor_score": stopper.anchor_score},
           "metric_calls": ad.calls, "reflections": teacher.usage, "seconds": round(time.time() - t0, 1),
           "in_sample": True, "runs": 0, "score": scores[best] if scores else None, "per_task": {}, "rules": {}}
    with open(os.path.join(put_paths.RUNS, f"{label}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, default=str)
    import put_telemetry
    print(f"telemetry: {put_telemetry.push(doc)} gauge lines pushed", flush=True)
    text = put_report.markdown(doc, contract)
    print(text)
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(text)
    return 0
