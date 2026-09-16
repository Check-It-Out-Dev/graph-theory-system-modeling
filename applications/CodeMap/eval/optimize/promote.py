"""The promotion gate: a candidate becomes the next navigator template only when the run says so.

    python eval/optimize/promote.py --run eval/optimize/runs/<date>.json [--candidate <file>] [--apply] [--pr]

Gate (all must hold): the run is a GEPA run, `win` is true, the best candidate beats the seed on the
validation set by at least MIN_GAIN, it satisfies `constraints.check`, and it is not byte-identical
to the current template. `--apply` writes `prompts/navigator/template.md`, builds the next immutable
`v<N>.md` and `active.md` with the builder, and appends a PROMPT_LOG row; `--pr` additionally commits
on a branch and opens a pull request (the owner merges — promotion is a proposal too). Without
`--apply` the script only prints the decision, which is the recorded outcome of a no-win run.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import constraints  # noqa: E402

MIN_GAIN = 0.03
NAV = os.path.join(R, "prompts", "navigator")


def sha16(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def next_version():
    n = 1
    for name in os.listdir(NAV):
        if name.startswith("v") and name.endswith(".md") and name[1:-3].isdigit():
            n = max(n, int(name[1:-3]) + 1)
    return n


def decide(run, candidate_text, current_text):
    reasons = []
    if run.get("mode") != "gepa":
        reasons.append(f"run mode is {run.get('mode')}, not gepa")
    if not run.get("win"):
        reasons.append("the run recorded no win")
    seed, best = run.get("seed_val_score"), run.get("best_val_score")
    if seed is None or best is None or best - seed < MIN_GAIN:
        reasons.append(f"gain {None if seed is None or best is None else round(best - seed, 4)} < {MIN_GAIN}")
    probs = constraints.check(candidate_text)
    if probs:
        reasons.append("constraints: " + "; ".join(probs))
    if candidate_text.replace("\r\n", "\n") == current_text.replace("\r\n", "\n"):
        reasons.append("candidate is byte-identical to the current template")
    return (not reasons), reasons


def apply(run, candidate_text, pr=False, base=None):
    version = next_version()
    candidate_text = candidate_text.replace("\r\n", "\n").rstrip("\n") + "\n"
    with open(os.path.join(NAV, "template.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(candidate_text)
    build = os.path.join(R, "tools", "prompt", "build_navigator.py")
    subprocess.run([sys.executable, build, "--version", str(version)], check=True)  # NOSONAR - argv list of our own script
    subprocess.run([sys.executable, build, "--out", os.path.join(NAV, "active.md")], check=True)  # NOSONAR
    row = (f"| v{version} | {time.strftime('%Y-%m-%d')} | v{version - 1} | {sha16(candidate_text)} | GEPA run {run.get('date')}: "
           f"val {run.get('seed_val_score')} → {run.get('best_val_score')} on {len(run.get('val', []))} examples, "
           f"{run.get('metric_calls')} metric calls | promoted ({MIN_GAIN} gate) |")
    with open(os.path.join(NAV, "PROMPT_LOG.md"), "a", encoding="utf-8", newline="\n") as f:
        f.write(row + "\n")
    print(f"promoted: template.md, v{version}.md, active.md, PROMPT_LOG row")
    if pr:
        br = f"prompt/v{version}"
        base = base or subprocess.run(["git", "-C", R, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()  # NOSONAR
        subprocess.run(["git", "-C", R, "checkout", "-b", br], check=True)  # NOSONAR
        subprocess.run(["git", "-C", R, "add", "prompts/navigator", "eval/optimize/runs"], check=True)  # NOSONAR
        subprocess.run(["git", "-C", R, "commit", "-m", f"Navigator prompt v{version}: GEPA run {run.get('date')} ({run.get('seed_val_score')} → {run.get('best_val_score')})"], check=True)  # NOSONAR
        subprocess.run(["git", "-C", R, "push", "-u", "origin", br], check=True)  # NOSONAR
        subprocess.run(["gh", "pr", "create", "--fill", "--head", br, "--base", base], check=True)  # NOSONAR
        subprocess.run(["git", "-C", R, "checkout", base], check=True)  # NOSONAR
    return version


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--pr", action="store_true")
    ap.add_argument("--base", default=None, help="pull request base (default: the current branch)")
    a = ap.parse_args(argv)
    run = json.load(open(a.run, encoding="utf-8"))
    cand_path = a.candidate or a.run.replace(".json", ".template.md")
    if not os.path.exists(cand_path):
        print(f"no candidate file ({cand_path}); nothing to promote")
        return 0
    candidate = open(cand_path, encoding="utf-8").read()
    current = open(os.path.join(NAV, "template.md"), encoding="utf-8").read()
    ok, reasons = decide(run, candidate, current)
    print(json.dumps({"promote": ok, "reasons": reasons, "seed": run.get("seed_val_score"), "best": run.get("best_val_score")}, indent=1))
    if ok and a.apply:
        apply(run, candidate, pr=a.pr, base=a.base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
