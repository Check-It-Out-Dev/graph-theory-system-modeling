"""The markdown report of a campaign summary: what a reviewer reads in the Actions job summary and in the repository.

    markdown(summary, contract) -> str
"""

RULE_ORDER = ("hidden_pass", "build_green", "tests_written", "graph_first", "exemplar_read", "feature_first",
              "constructor_injection", "after_commit_listener", "scheduler_lock", "ports_adapters", "translatable_errors",
              "liquibase_changeset", "version_field", "controller_guards", "dto_naming", "scope", "marker")


def _pct(v):
    return "—" if v is None else f"{v:.2f}"


def rule_table(rules, bound):
    lines = ["| rule | passed / applicable | rate | 95 % Wilson | claim |", "|---|---|---|---|---|"]
    for rid in RULE_ORDER:
        r = rules.get(rid)
        if not r or not r["n"]:
            continue
        claim = "obligatory" if r["obligatory"] else ("below the bound" if r["n"] >= 35 else "reported-only (n < 35)")
        lines.append(f"| {rid} | {r['passed']} / {r['n']} | {_pct(r['rate'])} | [{r['wilson'][0]:.2f}, {r['wilson'][1]:.2f}] | {claim} |")
    lines.append(f"\nA run passes a rule only at value 1. `obligatory` = lower Wilson bound ≥ {bound}; with no failure that needs 35 runs.")
    return "\n".join(lines)


def markdown(summary, contract):
    s = summary
    out = [f"## ◇ Prompt under test — `{s['label']}` ({s['mode']})", "",
           f"Instance `{s['instance']}` at `{s['base_sha'][:10]}`; prompt `{s['prompt_version']}` ({s.get('prompt_origin', '')}); "
           f"coder `{s['models']['coder']}`, judge `{s['models']['judge']}`; {s['runs']} runs, {s.get('unjudged', 0)} without a verdict.", ""]
    if s.get("score") is not None:
        out.append(f"**Score {s['score']:.3f}** (mean over tasks of the mean over replicates).")
    if s.get("noise"):
        n = s["noise"]
        out.append(f"Noise floor **δ = {n['delta']:.3f}** = max(judge test-retest MAD {n['judge_mad_score']:.3f}, "
                   f"agent half-width {n['agent_halfwidth']:.3f} from pooled SD {n['agent_pooled_sd']:.3f} over {n['tasks']} tasks × k={n['k']}).")
    if s.get("stop"):
        st = s["stop"]
        out.append(f"Stopped: **{st.get('reason')}** after iteration {st.get('after_iteration')} (δ {st.get('delta')}, streak {st.get('streak')}).")
    out += ["", "### Per task", "", "| task | scores | mean |", "|---|---|---|"]
    for t, v in sorted(s.get("per_task", {}).items()):
        out.append(f"| {t} | {', '.join(f'{x:.3f}' for x in v['scores'])} | {v['mean']:.3f} |")
    out += ["", "### Per rule", "", rule_table(s.get("rules", {}), contract["statistics"]["obligatory_lower_bound"]), ""]
    if s.get("untaught_rules"):
        out += ["### Rules the prompt could not teach", "", "| rule | rate | recommended enforcement |", "|---|---|---|"]
        for r in s["untaught_rules"]:
            out.append(f"| {r['rule']} | {_pct(r.get('rate'))} | {r.get('enforce', 'a CI check or a pre-commit hook')} |")
        out.append("")
    return "\n".join(out) + "\n"
