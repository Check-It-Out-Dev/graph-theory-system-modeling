"""Lint the dashboards before they are provisioned: valid JSON, no template variables, unique explicit
panel ids, `${DS_PROMETHEUS}` declared, every counter query wrapped in max_over_time (public panels
see running totals), no currency anywhere. Exit 1 on the first family of problems.

    python tools/grafana/lint.py [observability/grafana/*.json]
"""

import glob
import json
import os
import re
import sys

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COUNTERS = ("codemap_requests_total", "codemap_credits_total", "codemap_tokens_total", "codemap_feedback_total",
            "codemap_feedback_tags_total", "codemap_miss_total", "codemap_budget_refusals_total", "codemap_steps_total",
            "codemap_feedback_verified_total")
CURRENCY = re.compile(r"\b(usd|eur|pln|\$\d|price|cost_usd)\b", re.I)


def lint_doc(doc, name):
    problems = []
    if doc.get("templating", {}).get("list"):
        problems.append(f"{name}: template variables are not allowed on public dashboards")
    if not any(i.get("name") == "DS_PROMETHEUS" for i in doc.get("__inputs", [])):
        problems.append(f"{name}: __inputs must declare DS_PROMETHEUS")
    ids = []

    def walk(panels):
        for p in panels:
            if "id" not in p:
                problems.append(f"{name}: panel without id: {p.get('title')}")
            ids.append(p.get("id"))
            for t in p.get("targets", []) or []:
                expr = t.get("expr", "")
                for c in COUNTERS:
                    if re.search(rf"\b{c}\b", expr) and "max_over_time(" not in expr and "rate(" not in expr:
                        problems.append(f"{name}: panel {p.get('id')} queries counter {c} without max_over_time/rate: {expr}")
                if "$" in expr and "$__range" not in expr and "$__interval" not in expr:
                    problems.append(f"{name}: panel {p.get('id')} uses a variable: {expr}")
            if p.get("type") == "text" and CURRENCY.search(p.get("options", {}).get("content", "")):
                problems.append(f"{name}: panel {p.get('id')} text mentions currency")
            if CURRENCY.search(json.dumps(p.get("title", "")) + json.dumps(p.get("description", ""))):
                problems.append(f"{name}: panel {p.get('id')} title/description mentions currency")
            walk(p.get("panels", []) or [])
    walk(doc.get("panels", []))
    if len(ids) != len(set(ids)):
        problems.append(f"{name}: duplicate panel ids")
    if not doc.get("uid", "").startswith("codemap-"):
        problems.append(f"{name}: uid must start with codemap-")
    return problems


def main(argv=None):
    paths = argv or sorted(glob.glob(os.path.join(R, "observability", "grafana", "*.json")))
    if not paths:
        print("no dashboards found")
        return 1
    problems = []
    for p in paths:
        try:
            doc = json.load(open(p, encoding="utf-8"))
        except ValueError as e:
            problems.append(f"{p}: invalid JSON: {e}")
            continue
        problems += lint_doc(doc, os.path.basename(p))
    for pr in problems:
        print(pr)
    print(f"{len(paths)} dashboards, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
