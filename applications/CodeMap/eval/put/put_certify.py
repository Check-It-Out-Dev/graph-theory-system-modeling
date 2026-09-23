"""Certification: the seed and a candidate on every task, replicated, and the verdict the claims rest on.

    PYTHONUTF8=1 python eval/put/run.py certify --campaign <campaign.json> --base-repo <checkout>
      campaign: {"label", "candidate": <body file, e.g. runs/<gepa label>/best_prompt.md>, "gepa": <gepa label>,
                 "baseline": <baseline label>, "reuse": [labels whose finished cells may be copied]}

The seed runs k_certify_seed times and the candidate k_certify_promoted times on all ten tasks. Finished cells of the
same prompt on the same task under an earlier label (the baseline's seed runs, GEPA's runs of the candidate) are
copied instead of rerun: same prompt, same task, same base commit, same models.

The verdict (METRICS.md section 1e): the candidate works when its mean score on the hold-out tasks exceeds the
seed's by more than delta, the paired-bootstrap 95 % CI of that gain excludes 0, and no rule obligatory for the seed
stops being obligatory. The ceiling (section 1d): every rule that applies to every task is obligatory for the
candidate (lower Wilson bound >= 0.90 over its 40 runs) -> saturated_at_ceiling; otherwise the rules below the bound
are the rules the prompt could not teach.
"""

import json
import os
import shutil
import time
from statistics import mean

import put_paths
import put_cli
import put_checks
import put_contract
import put_prompt
import put_report
import put_runner
import put_score
import put_stats

UNIVERSAL_SKIP = ("hidden_pass",)          # correctness is the task's, not a convention the prompt teaches


def reuse_cells(from_labels, label, body, tasks, reps):
    sha = put_prompt.sha16(body)
    copied = 0
    for src_label in from_labels or ():
        for t in tasks:
            for r in range(reps):
                src = os.path.join(put_paths.RUNS, src_label, sha, f"{t['id']}.r{r}")
                dst = os.path.join(put_paths.RUNS, label, sha, f"{t['id']}.r{r}")
                if put_runner.finished(src) and not put_runner.finished(dst):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    copied += 1
    return copied


def split_scores(records, split):
    return {t: v for t, v in put_score.by_task([r for r in records if r["split"] == split]).items()}


def verdict(seed_recs, cand_recs, contract, delta):
    st = contract["statistics"]
    out = {}
    for split in ("train", "holdout"):
        s, c = split_scores(seed_recs, split), split_scores(cand_recs, split)
        out[split] = {"seed": round(mean(mean(v) for v in s.values()), 4) if s else None,
                      "candidate": round(mean(mean(v) for v in c.values()), 4) if c else None}
    a, b = split_scores(seed_recs, "holdout"), split_scores(cand_recs, "holdout")
    point, lo, hi = put_stats.paired_bootstrap(a, b, st["bootstrap_resamples"], st["bootstrap_seed"])
    x, n, p = put_stats.sign_test([mean(b[t]) - mean(a[t]) for t in sorted(set(a) & set(b))])
    out["gain_holdout"] = {"mean": round(point, 4), "ci": [round(lo, 4), round(hi, 4)], "delta": delta,
                           "sign_test": {"positive": x, "n": n, "p_one_sided": round(p, 4)}}
    rules_seed = put_score.rule_rates(seed_recs, contract)
    rules_cand = put_score.rule_rates(cand_recs, contract)
    lost = [r for r, v in rules_seed.items() if v["obligatory"] and not rules_cand.get(r, {}).get("obligatory")]
    works = point > delta and lo > 0 and not lost
    out["works"] = works
    out["obligatory_lost"] = lost
    out["why"] = ("the hold-out gain exceeds delta, its CI excludes 0 and no obligatory rule was lost" if works else
                  "; ".join(filter(None, [
                      f"hold-out gain {point:.3f} is not above delta {delta:.3f}" if point <= delta else "",
                      f"the CI [{lo:.3f}, {hi:.3f}] does not exclude 0" if lo <= 0 else "",
                      f"obligatory rules lost: {', '.join(lost)}" if lost else ""])))
    return out, rules_seed, rules_cand


def ceiling(rules_cand, contract):
    universal = [r["id"] for r in contract["rules"] if r["kind"] == "deterministic" and r.get("applies") == "all"
                 and r["id"] not in UNIVERSAL_SKIP]
    untaught = [{"rule": r, "rate": rules_cand[r]["rate"], "wilson": rules_cand[r]["wilson"],
                 "enforce": ENFORCE.get(r, "a CI check or a pre-commit hook")}
                for r in universal if r in rules_cand and not rules_cand[r]["obligatory"]]
    specific = [{"rule": r, "rate": v["rate"], "n": v["n"], "wilson": v["wilson"]} for r, v in rules_cand.items()
                if r not in universal and r not in UNIVERSAL_SKIP and v["n"] and (v["rate"] or 0) < contract["statistics"]["obligatory_lower_bound"]]
    return ("saturated_at_ceiling" if not untaught else "saturated_below_ceiling"), untaught, specific


ENFORCE = {
    "graph_first": "a session hook that blocks the first Edit or Write until the graph tool was called",
    "exemplar_read": "a session hook that asks for an example file before the first new class",
    "feature_first": "an ArchUnit rule on package placement",
    "constructor_injection": "an ArchUnit rule (no field injection) or PMD's field-injection rule",
    "tests_written": "a CI gate: a change under src/main needs a changed or new *UnitTest",
    "build_green": "the existing pre-merge build",
    "scope": "a CI diff-size and path gate",
    "marker": "the harness: end the session on the answer contract",
}


def main_from_cli(camp, label, contract, instance, base_repo, parallel, summary_path):
    st = contract["statistics"]
    repo = put_checks.Repo(base_repo, contract["base_sha"])
    tasks = put_contract.tasks(instance)
    seed_body = put_prompt.load_body(instance, camp.get("seed", "v1"))
    with open(camp["candidate"], encoding="utf-8") as f:
        cand_body = f.read()
    with open(os.path.join(put_paths.RUNS, f"{camp['baseline']}.json"), encoding="utf-8") as f:
        delta = json.load(f)["noise"]["delta"]
    reused = {"seed": reuse_cells(camp.get("reuse"), label, seed_body, tasks, st["k_certify_seed"]),
              "candidate": reuse_cells(camp.get("reuse"), label, cand_body, tasks, st["k_certify_promoted"])}
    put_cli.log(f"reused finished cells: {reused}")
    for body, name in ((seed_body, "seed"), (cand_body, "candidate")):
        ok, detail = put_cli.preflight(body, instance)
        put_cli.log(f"preflight {name}: {detail}")
        if not ok:
            return 3
    seed_cells = [(t, r) for r in range(st["k_certify_seed"]) for t in tasks]
    cand_cells = [(t, r) for r in range(st["k_certify_promoted"]) for t in tasks]
    seed_dirs = put_cli.run_cells(seed_cells, seed_body, label, contract, instance, base_repo, parallel)
    cand_dirs = put_cli.run_cells(cand_cells, cand_body, label, contract, instance, base_repo, parallel)
    seed_recs = put_cli.assess(seed_dirs, contract, instance, repo)
    cand_recs = put_cli.assess(cand_dirs, contract, instance, repo)
    v, rules_seed, rules_cand = verdict(seed_recs, cand_recs, contract, delta)
    reason, untaught, specific = ceiling(rules_cand, contract)
    extra = {"seed_version": put_prompt.version(seed_body), "candidate_version": put_prompt.version(cand_body),
             "candidate_file": camp["candidate"], "gepa": camp.get("gepa"), "baseline": camp["baseline"],
             "reused": reused, "verdict": v, "rules_seed": rules_seed, "rules_candidate": rules_cand,
             "ceiling": reason, "untaught_rules": untaught, "below_bound_specific": specific,
             "seed_runs": len(seed_recs), "candidate_runs": len(cand_recs),
             "seed_records": [{k: r[k] for k in ("task", "split", "rep", "score")} for r in seed_recs]}
    summary, path = put_cli.summarize(label, "certify", cand_body, camp["candidate"], cand_recs, contract, extra)
    put_cli.log(f"telemetry: {put_cli.put_telemetry.push(summary)} gauge lines pushed")
    text = put_report.markdown(summary, contract) + certify_markdown(summary)
    put_cli.log(f"certification {path}: works={v['works']} ({v['why']}); ceiling {reason}")
    print(text)
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(text)
    return 0


def certify_markdown(s):
    v = s["verdict"]
    g = v["gain_holdout"]
    lines = ["### Certification", "",
             "| | seed | candidate |", "|---|---|---|",
             f"| training tasks (in-sample, best-of-k) | {v['train']['seed']} | {v['train']['candidate']} |",
             f"| hold-out tasks | {v['holdout']['seed']} | {v['holdout']['candidate']} |", "",
             f"Hold-out gain **{g['mean']:.3f}** (95 % CI {g['ci'][0]:.3f} to {g['ci'][1]:.3f}; δ {g['delta']:.3f}); "
             f"sign test {g['sign_test']['positive']}/{g['sign_test']['n']} tasks, p = {g['sign_test']['p_one_sided']}.",
             f"**The prompt works: {'yes' if v['works'] else 'no'}** — {v['why']}.",
             f"Ceiling: **{s['ceiling']}**.", ""]
    rs, rc = s["rules_seed"], s["rules_candidate"]
    lines += ["| rule | seed rate [Wilson] | candidate rate [Wilson] | candidate claim |", "|---|---|---|---|"]
    for rid in put_report.RULE_ORDER:
        a, b = rs.get(rid), rc.get(rid)
        if not a or not b or not b["n"]:
            continue
        claim = "obligatory" if b["obligatory"] else ("reported-only" if b["n"] < 35 else "below the bound")
        lines.append(f"| {rid} | {put_report._pct(a['rate'])} [{a['wilson'][0]:.2f}, {a['wilson'][1]:.2f}] (n {a['n']}) | "
                     f"{put_report._pct(b['rate'])} [{b['wilson'][0]:.2f}, {b['wilson'][1]:.2f}] (n {b['n']}) | {claim} |")
    return "\n".join(lines) + "\n"
