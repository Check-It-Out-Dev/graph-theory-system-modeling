"""Promotion of a certified prompt: the gate, then the CLAUDE.md a team would distribute, on a branch of the repository
under test (the owner merges; nothing here merges or pushes to a default branch).

    PYTHONUTF8=1 python eval/put/put_promote.py decide --certify <label>
    PYTHONUTF8=1 python eval/put/put_promote.py apply  --certify <label> --target-repo <checkout> [--branch B] [--push]

The gate (all must hold): the GEPA run stopped on its plateau or its budget; the certification says the prompt works
(hold-out gain above max(min_gain_floor, delta) with a CI excluding 0, no obligatory rule lost); at least
min_examples_for_promotion held-out and validation examples; the candidate differs from the current prompt.
A failed gate is a result too: the report says which condition failed, and the seed stays the team's prompt.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import put_paths
import put_contract
import put_prompt

PREFACE = ("<!-- Distributed by the prompt-under-test pipeline (graph-theory-system-modeling, applications/CodeMap/eval/put). "
           "Version {version}, certified by run {label}. Edit through the pipeline, not by hand: a change is measured before it ships. -->\n"
           "If this session has no `graph` tool, skip the code-graph steps below and find files with search instead.\n\n")


def load(label):
    with open(os.path.join(put_paths.RUNS, f"{label}.json"), encoding="utf-8") as f:
        return json.load(f)


def decide(cert, contract, gepa=None, current_body=None, candidate_body=None):
    st = contract["statistics"]
    v = cert["verdict"]
    reasons = []
    floor = max(st["min_gain_floor"], v["gain_holdout"]["delta"])
    if v["gain_holdout"]["mean"] <= floor:
        reasons.append(f"hold-out gain {v['gain_holdout']['mean']:.3f} is not above max(floor {st['min_gain_floor']}, delta {v['gain_holdout']['delta']:.3f})")
    if v["gain_holdout"]["ci"][0] <= 0:
        reasons.append(f"the gain's CI lower end {v['gain_holdout']['ci'][0]:.3f} is not above 0")
    if v["obligatory_lost"]:
        reasons.append("obligatory rules lost: " + ", ".join(v["obligatory_lost"]))
    # examples = the candidate's held-out runs of the certification + the validation tasks GEPA scored it on
    n_holdout = sum(1 for r in cert.get("records", []) if r.get("split") == "holdout")
    n_val = len([t for t in put_contract.tasks(contract["instance"]) if t["split"] == "train"]) if gepa else 0
    n_examples = n_holdout + n_val
    if n_examples < st["min_examples_for_promotion"]:
        reasons.append(f"{n_examples} examples, fewer than {st['min_examples_for_promotion']}")
    if gepa and gepa.get("stop", {}).get("reason") not in ("plateau", "budget"):
        reasons.append(f"the GEPA run stopped by {gepa.get('stop', {}).get('reason')}, not on its plateau or budget")
    if current_body is not None and candidate_body is not None and current_body.strip() == candidate_body.strip():
        reasons.append("the candidate is the current prompt")
    return (not reasons), reasons


def render_for_repo(body, instance, label):
    text = put_prompt.render(body, instance)
    return PREFACE.format(version=put_prompt.version(body), label=label) + text


def apply(cert_label, instance, target_repo, branch=None, push=False, base="origin/main"):
    cert = load(cert_label)
    with open(cert["candidate_file"], encoding="utf-8") as f:
        body = f.read()
    version = put_prompt.version(body)
    branch = branch or f"prompt/backend-conventions-{version.split('@')[1][:8]}"
    wt = os.path.join(os.path.expanduser("~"), "put-ws", "promote-" + branch.replace("/", "-"))
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=target_repo, check=True)
    if not os.path.exists(wt):
        subprocess.run(["git", "worktree", "add", "-B", branch, wt, base], cwd=target_repo, check=True)
    with open(os.path.join(wt, "CLAUDE.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(render_for_repo(body, instance, cert_label))
    log_row = (f"| {time.strftime('%Y-%m-%d')} | {version} | {cert_label} | hold-out gain {cert['verdict']['gain_holdout']['mean']:.3f} "
               f"(CI {cert['verdict']['gain_holdout']['ci'][0]:.3f} to {cert['verdict']['gain_holdout']['ci'][1]:.3f}) | {cert['ceiling']} |\n")
    plog = os.path.join(put_paths.instance_dir(instance), "PROMPT_LOG.md")
    if not os.path.exists(plog):
        with open(plog, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Prompt log — backend-conventions\n\nEvery version that reached the repository, with the run that certified it.\n\n"
                    "| date | version | certified by | hold-out result | ceiling |\n|---|---|---|---|---|\n")
    with open(plog, "a", encoding="utf-8", newline="\n") as f:
        f.write(log_row)
    subprocess.run(["git", "add", "CLAUDE.md"], cwd=wt, check=True)
    msg = (f"CLAUDE.md: the team's conventions prompt, certified by the prompt-under-test pipeline ({version})\n\n"
           f"Certified by run {cert_label} in graph-theory-system-modeling (applications/CodeMap/eval/put): hold-out gain "
           f"{cert['verdict']['gain_holdout']['mean']:.3f}, ceiling {cert['ceiling']}. Every coding agent working in this "
           f"repository reads it as its project instructions.\n\nCo-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>\n")
    subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=wt, input=msg.encode("utf-8"), check=True)
    if push:
        subprocess.run(["git", "push", "-q", "-u", "origin", branch], cwd=wt, check=True)
    return wt, branch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["decide", "apply"])
    ap.add_argument("--certify", required=True)
    ap.add_argument("--gepa")
    ap.add_argument("--instance", default="backend-conventions")
    ap.add_argument("--target-repo")
    ap.add_argument("--branch")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    contract = put_contract.load(a.instance)
    cert = load(a.certify)
    gepa = load(a.gepa) if a.gepa else None
    with open(cert["candidate_file"], encoding="utf-8") as f:
        cand = f.read()
    ok, reasons = decide(cert, contract, gepa, put_prompt.load_body(a.instance, "v1"), cand)
    print(json.dumps({"promote": ok, "reasons": reasons}, indent=1))
    if a.cmd == "apply":
        if not ok:
            print("the gate is closed; nothing applied")
            return 2
        wt, branch = apply(a.certify, a.instance, a.target_repo, a.branch, a.push)
        print(f"committed CLAUDE.md on {branch} in {wt}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
