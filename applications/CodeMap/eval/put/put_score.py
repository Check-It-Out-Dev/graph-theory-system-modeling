"""The score of one run and the aggregates of a campaign, as `contract.json` declares them.

    S(run) = sum_i w_i * m_i, every m_i in [0, 1]
      hidden_pass      deterministic: hidden acceptance + pass-to-pass tests passed / total (0 on a red build)
      conventions_det  deterministic: mean of the applicable convention rules of the task
      tests_written    deterministic: the agent's own unit tests (0 on a red build)
      process_det      deterministic: mean of graph_first and exemplar_read
      correctness, convention_fit, design_fit, test_quality, graph_use: judge r4, each (s - 1) / 4

A run without a valid verdict scores its judge part as 0 and is flagged, never silently dropped.
"""

from statistics import mean

import put_stats


def judge_part(verdict, criterion):
    if not verdict or verdict.get(criterion) is None:
        return 0.0
    return (float(verdict[criterion]) - 1.0) / 4.0


def components(checks, verdict, contract):
    """-> {weight key: m in [0, 1]} for one run."""
    c = checks["checks"] if "checks" in checks else checks
    groups = contract["weight_groups"]
    build = bool(c.get("build_green", {}).get("value"))
    out = {}
    for key in contract["weights"]:
        if key in groups:
            vals = [c[r]["value"] for r in groups[key] if r in c and c[r].get("applicable") and c[r].get("value") is not None]
            out[key] = mean(vals) if vals else 1.0
        elif key in contract["judge"]["criteria"]:
            out[key] = judge_part(verdict, key)
        else:
            v = c.get(key, {}).get("value")
            out[key] = float(v) if v is not None else 0.0
    if not build:                                                     # the gate: a red build earns no test credit
        out["hidden_pass"] = 0.0
        out["tests_written"] = 0.0
    return out


def score(checks, verdict, contract):
    comp = components(checks, verdict, contract)
    return round(sum(contract["weights"][k] * comp[k] for k in contract["weights"]), 6), comp


def rule_rates(records, contract, z=1.96, bound=None):
    """records: [{"task", "split", "checks"}]. -> {rule: {"passed", "n", "rate", "wilson", "obligatory"}} over
    the runs the rule applies to; a run passes a rule only at value 1."""
    bound = bound if bound is not None else contract["statistics"]["obligatory_lower_bound"]
    out = {}
    for rule in contract["rules"]:
        if rule["kind"] != "deterministic":
            continue
        vals = [r["checks"][rule["id"]] for r in records if rule["id"] in r["checks"] and r["checks"][rule["id"]].get("applicable")]
        vals = [v for v in vals if v.get("value") is not None]
        k = sum(1 for v in vals if v.get("passed"))
        n = len(vals)
        lo, hi = put_stats.wilson(k, n, z)
        out[rule["id"]] = {"passed": k, "n": n, "rate": (k / n) if n else None, "wilson": [round(lo, 4), round(hi, 4)],
                           "obligatory": put_stats.obligatory(k, n, bound, z), "mean_value": mean(v["value"] for v in vals) if vals else None}
    return out


def by_task(records, key="score"):
    out = {}
    for r in records:
        if r.get(key) is not None:
            out.setdefault(r["task"], []).append(r[key])
    return out


def candidate_score(records, key="score"):
    """Mean over tasks of the mean over replicates."""
    groups = by_task(records, key)
    return mean(mean(v) for v in groups.values()) if groups else None
