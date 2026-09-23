"""The prompt-under-test campaigns: one command per mode, the same on the self-hosted runner and on a laptop.

    PYTHONUTF8=1 python eval/put/run.py <mode> --campaign <campaign.json> --base-repo <checkout> [--label L] [--summary F]

modes
    smoke      the prompt under test (the repository's CLAUDE.md, else the seed) on 2 training tasks, once each
    baseline   the seed on every training task k times; judged twice; the noise floor delta and per-rule rates
    gepa       GEPA over the prompt until the plateau stopper or the metric-call budget ends it (put_gepa)
    certify    seed k_certify_seed and a candidate k_certify_promoted times on all tasks; the hold-out verdict
    report     rebuild the summary and the report of a label from its run directories (no model call)

A campaign.json may set: instance, label, tasks ("train" | "holdout" | "all" | [ids]), k, prompt ("v1", a path to a
body file, or "repo" for <base-repo>/CLAUDE.md), candidate (a body file, certify), parallel, max_metric_calls.
Every cell (prompt sha16, task, replicate) is cached by its run directory: a finished cell is never rerun (law 7).
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean

import put_paths
import put_checks
import put_contract
import put_judge
import put_prompt
import put_report
import put_runner
import put_score
import put_stats
import run_pairs


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_campaign(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_prompt(spec, instance, base_repo):
    """-> (body, origin). 'v1' = a committed version; 'repo' = the repository's CLAUDE.md (seed if absent); else a file."""
    if spec in (None, "", "seed"):
        spec = "v1"
    if spec == "repo":
        p = os.path.join(base_repo, "CLAUDE.md")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return f.read(), "repo CLAUDE.md"
        return put_prompt.load_body(instance, "v1"), "seed v1 (the repository has no CLAUDE.md)"
    if os.path.exists(os.path.join(put_paths.instance_dir(instance), "prompt", f"{spec}.md")):
        return put_prompt.load_body(instance, spec), f"prompt/{spec}.md"
    with open(spec, encoding="utf-8") as f:
        return f.read(), spec


def select_tasks(instance, spec):
    rows = put_contract.tasks(instance)
    if spec in (None, "train", "holdout"):
        return [t for t in rows if t["split"] == (spec or "train")]
    if spec == "all":
        return rows
    return [t for t in rows if t["id"] in spec]


def preflight(body, instance):
    rendered = put_prompt.render(body, instance)
    tmp = tempfile.mkdtemp(prefix="put-preflight-")
    ok, detail = run_pairs.preflight({"context": put_runner.context_dir(tmp, rendered)}, put_prompt.checksum(rendered), cwd=tmp)
    return ok, detail


def run_cells(cells, body, label, contract, instance, base_repo, parallel):
    """cells: [(task, rep)]. Runs the unfinished ones; returns their run directories (all cells)."""
    dirs = []
    todo = []
    for task, rep in cells:
        rd = put_runner.run_dir_for(put_paths.RUNS, label, body, task["id"], rep)
        dirs.append((task, rd))
        if not put_runner.finished(rd):
            todo.append((task, rd))
    log(f"{len(cells)} cells, {len(cells) - len(todo)} cached, {len(todo)} to run (parallel {parallel})")

    def one(task, rd):
        t0 = time.time()
        try:
            meta = put_runner.execute(task, body, rd, base_repo, contract, instance, label=label)
            log(f"done {task['id']} r{meta['rep']} in {time.time() - t0:.0f}s marker={meta['marker']} "
                f"turns={meta['usage'].get('num_turns')} budget={meta['session'].get('budget_exhausted')}")
        except Exception as ex:                                        # a failed cell is visible and can be rerun
            log(f"FAILED {task['id']} {rd}: {type(ex).__name__}: {ex}")

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(one, t, rd) for t, rd in todo]
        for _ in as_completed(futures):
            pass
    return dirs


def assess(dirs, contract, instance, repo, judge=True, repeat=False, judge_parallel=3):
    """Checks (recomputed), the verdict (cached), optionally a second verdict in shuffled order. -> records."""
    records = []
    for task, rd in dirs:
        if not put_runner.finished(rd):
            continue
        checks = put_checks.check_run(rd, task, contract, repo)
        records.append({"task": task["id"], "split": task["split"], "rep": int(rd.rsplit(".r", 1)[-1]), "run_dir": rd,
                        "checks": checks["checks"]})
    if judge:
        def grade(rec, name):
            task = next(t for t, d in dirs if d == rec["run_dir"])
            return put_judge.judge_run(rec["run_dir"], task, instance, contract["models"]["judge"], name=name)
        with ThreadPoolExecutor(max_workers=judge_parallel) as pool:
            for rec, v in zip(records, pool.map(lambda r: grade(r, "verdict-r4.json"), records)):
                rec["verdict"] = v.get("verdict")
        if repeat:
            shuffled = records[:]
            random.Random(7).shuffle(shuffled)
            with ThreadPoolExecutor(max_workers=judge_parallel) as pool:
                for rec, v in zip(shuffled, pool.map(lambda r: grade(r, "verdict-r4-repeat.json"), shuffled)):
                    rec["verdict_repeat"] = v.get("verdict")
    for rec in records:
        rec["score"], rec["components"] = put_score.score({"checks": rec["checks"]}, rec.get("verdict"), contract)
        if rec.get("verdict_repeat") is not None:
            rec["score_repeat"], _ = put_score.score({"checks": rec["checks"]}, rec["verdict_repeat"], contract)
    return records


def noise(records, contract):
    z = contract["statistics"]["z"]
    judge_mad = put_stats.mad([(r["score"], r.get("score_repeat")) for r in records])
    crit_mad = {c: put_stats.mad([((r.get("verdict") or {}).get(c), (r.get("verdict_repeat") or {}).get(c)) for r in records])
                for c in contract["judge"]["criteria"]}
    groups = put_score.by_task(records)
    pooled = put_stats.pooled_sd(list(groups.values()))
    k = max((len(v) for v in groups.values()), default=0)
    d, agent = put_stats.delta(judge_mad, pooled, len(groups), k, z)
    return {"judge_mad_score": round(judge_mad, 4), "judge_mad_criteria": {c: round(v, 3) for c, v in crit_mad.items()},
            "agent_pooled_sd": round(pooled, 4), "agent_halfwidth": round(agent, 4), "tasks": len(groups), "k": k,
            "delta": round(d, 4), "rule": contract["statistics"]["delta"]}


def summarize(label, mode, body, origin, records, contract, extra=None):
    per_task = {t: {"scores": [round(s, 4) for s in v], "mean": round(mean(v), 4)} for t, v in put_score.by_task(records).items()}
    out = {"schema": 1, "label": label, "mode": mode, "instance": contract["instance"], "base_sha": contract["base_sha"],
           "prompt_version": put_prompt.version(body), "prompt_origin": origin,
           "models": contract["models"], "weights": contract["weights"],
           "runs": len(records), "score": round(put_score.candidate_score(records), 4) if records else None,
           "per_task": per_task, "rules": put_score.rule_rates(records, contract),
           "unjudged": sum(1 for r in records if r.get("verdict") is None),
           "records": [{k: v for k, v in r.items() if k not in ("run_dir",)} | {"run": os.path.relpath(r["run_dir"], put_paths.RUNS)}
                       for r in records],
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if extra:
        out.update(extra)
    path = os.path.join(put_paths.RUNS, f"{label}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    return out, path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "baseline", "gepa", "certify", "report"])
    ap.add_argument("--campaign")
    ap.add_argument("--base-repo", default="C:/Users/Norbert/erdos-ws/backend")
    ap.add_argument("--label")
    ap.add_argument("--summary", help="append the markdown report here (GITHUB_STEP_SUMMARY)")
    ap.add_argument("--parallel", type=int)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args(argv)
    camp = load_campaign(a.campaign)
    instance = camp.get("instance", "backend-conventions")
    contract = put_contract.load(instance)
    label = a.label or camp.get("label") or f"{a.mode}-{time.strftime('%Y%m%d-%H%M')}"
    base_repo = os.path.abspath(a.base_repo)
    parallel = a.parallel or camp.get("parallel", 3)
    repo = put_checks.Repo(base_repo, contract["base_sha"])
    st = contract["statistics"]
    log(f"campaign {label}: mode {a.mode}, instance {instance}, base repo {base_repo}")

    if a.mode == "gepa":
        import put_gepa
        return put_gepa.main_from_cli(camp, label, contract, instance, base_repo, parallel, a.summary)

    if a.mode == "certify":
        import put_certify
        return put_certify.main_from_cli(camp, label, contract, instance, base_repo, parallel, a.summary)

    if a.mode == "report":
        with open(os.path.join(put_paths.RUNS, f"{label}.json"), encoding="utf-8") as f:
            summary = json.load(f)
        text = put_report.markdown(summary, contract)
        print(text)
        if a.summary:
            with open(a.summary, "a", encoding="utf-8") as f:
                f.write(text)
        return 0

    body, origin = resolve_prompt(camp.get("prompt", "repo" if a.mode == "smoke" else "v1"), instance, base_repo)
    ok, detail = preflight(body, instance)
    log(f"preflight: {detail}")
    if not ok:
        log("the prompt did not reach the model in full; stopping (a CLAUDE.md can stay out of context silently)")
        return 3
    if a.mode == "smoke":
        tasks = select_tasks(instance, camp.get("tasks", ["stale-ticket-close", "faq-reorder"]))
        k = camp.get("k", 1)
    else:
        tasks = select_tasks(instance, camp.get("tasks", "train"))
        k = camp.get("k", st["k_baseline"])
    cells = [(t, r) for r in range(k) for t in tasks]
    dirs = run_cells(cells, body, label, contract, instance, base_repo, parallel)
    records = assess(dirs, contract, instance, repo, judge=not a.no_judge, repeat=(a.mode == "baseline"))
    extra = {"k": k, "tasks": [t["id"] for t in tasks], "preflight": detail}
    if a.mode == "baseline":
        extra["noise"] = noise(records, contract)
    summary, path = summarize(label, a.mode, body, origin, records, contract, extra)
    text = put_report.markdown(summary, contract)
    log(f"summary {path}: score {summary['score']} over {summary['runs']} runs")
    if a.summary:
        with open(a.summary, "a", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
