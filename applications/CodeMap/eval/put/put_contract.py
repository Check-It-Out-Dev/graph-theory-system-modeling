"""The contract of a prompt-under-test instance: its rules, their checks, the weights and the statistics settings,
and the tasks with their split. Declared before any run and committed; `validate` is the gate that keeps the
prompt, the contract and the tasks consistent with each other.

An instance directory holds:
    contract.json          rules (id -> kind, applies, predicate), weights, weight groups, statistics, runner settings
    prompt/v<N>.md         prompt versions; each `<rule id="...">` in the prompt must have a row in contract.rules
    prompt/references/     generated reference files the prompt includes
    tasks/tasks.jsonl      one task per line: id, split (train | holdout), tags, text, interface, exemplars, hidden, ...
    tasks/<id>/hidden/     the hidden acceptance tests (never shown to the coder, the judge or the reflector)
    tasks/<id>/reference.patch   a solution that the hidden tests accept (validates the tests, never shown)
"""

import json
import os
import re

import put_paths

RULE_TAG = re.compile(r'<rule id="([a-z_]+)">')
SPLITS = ("train", "holdout")


def load(instance):
    d = put_paths.instance_dir(instance)
    with open(os.path.join(d, "contract.json"), encoding="utf-8") as f:
        contract = json.load(f)
    contract["_dir"] = d
    return contract


def tasks(instance, split=None):
    d = put_paths.instance_dir(instance)
    with open(os.path.join(d, "tasks", "tasks.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [t for t in rows if split is None or t["split"] == split]


def prompt_text(instance, version="v1"):
    with open(os.path.join(put_paths.instance_dir(instance), "prompt", f"{version}.md"), encoding="utf-8") as f:
        return f.read()


def rule_ids(prompt):
    return RULE_TAG.findall(prompt or "")


def applies(rule, task):
    a = rule.get("applies", "all")
    return a == "all" or bool(set(a) & set(task.get("tags", ())))


def rules_for(contract, task):
    return [r for r in contract["rules"] if applies(r, task)]


def validate(contract, prompt, task_rows):
    """-> list of problems (empty when the instance is consistent)."""
    problems = []
    rules = {r["id"]: r for r in contract["rules"]}
    for rid in rule_ids(prompt):
        if rid not in rules:
            problems.append(f"prompt rule {rid} has no check in contract.rules")
    for rid, r in rules.items():
        if r.get("kind") not in ("deterministic", "judge"):
            problems.append(f"rule {rid}: kind must be deterministic or judge")
    for rid in rule_ids(prompt):
        if rule_ids(prompt).count(rid) > 1:
            problems.append(f"prompt rule {rid} appears more than once")
            break
    w = contract["weights"]
    if abs(sum(w.values()) - 1.0) > 1e-9:
        problems.append(f"weights sum to {sum(w.values())}, not 1")
    groups = contract.get("weight_groups", {})
    judged = set(contract["judge"]["criteria"])
    for key in w:
        if key not in rules and key not in groups and key not in judged:
            problems.append(f"weight {key} names no rule, group or judge criterion")
    for g, members in groups.items():
        for m in members:
            if m not in rules:
                problems.append(f"weight group {g} names unknown rule {m}")
    ids = [t["id"] for t in task_rows]
    if len(ids) != len(set(ids)):
        problems.append("task ids repeat")
    tags = {tag for t in task_rows for tag in t.get("tags", ())}
    rule_tags = {tag for r in rules.values() if r.get("applies") != "all" for tag in r["applies"]}
    for split in SPLITS:
        have = {tag for t in task_rows if t["split"] == split for tag in t.get("tags", ())}
        for tag in sorted(rule_tags):
            if tag not in have:
                problems.append(f"tag {tag} has no {split} task (coverage constraint)")
    for t in task_rows:
        if t["split"] not in SPLITS:
            problems.append(f"task {t['id']}: split {t['split']}")
        for key in ("text", "interface", "exemplars", "hidden", "package_prefix"):
            if not t.get(key):
                problems.append(f"task {t['id']}: {key} missing")
        for tag in t.get("tags", ()):
            if tag not in rule_tags:
                problems.append(f"task {t['id']}: tag {tag} selects no rule")
        if "schema" in t.get("tags", ()) and not (t.get("schema_table") and t.get("schema_column")):
            problems.append(f"task {t['id']}: a schema task names schema_table and schema_column")
        if "version" in t.get("tags", ()) and not t.get("contended_entities"):
            problems.append(f"task {t['id']}: a version task names contended_entities")
        if "vendor" in t.get("tags", ()) and not t.get("vendor"):
            problems.append(f"task {t['id']}: a vendor task names port_dir and adapter_dir")
        if "i18n" in t.get("tags", ()) and not t.get("i18n_keys"):
            problems.append(f"task {t['id']}: an i18n task names its message keys")
    if not tags:
        problems.append("no task has a tag")
    return problems
