"""Tasks of an instance: the card the coder sees, and what it never sees (hidden acceptance tests, the reference
solution), plus `validate`: every hidden test must fail on the base commit and pass on the reference solution,
together with the task's pass-to-pass classes (SWE-bench's FAIL_TO_PASS and PASS_TO_PASS).

    PYTHONUTF8=1 python eval/put/put_tasks.py validate --instance backend-conventions --source C:/Users/Norbert/erdos-ws/backend
    PYTHONUTF8=1 python eval/put/put_tasks.py reference --instance backend-conventions --task <id> --from <worktree>
"""

import argparse
import json
import os
import shutil
import sys
import time

import put_paths
import put_contract
import put_maven
import put_worktree

TEST_ROOT = os.path.join("src", "test", "java", "com", "sm", "instagram", "platform")

CARD = """TASK
{text}

INTERFACE
The reviewers' tests call the code through these names; keep them exactly.
{interface}

You work in a checkout of the checkItOut backend. Make the change in the working tree; do not commit."""


def card(task):
    return CARD.format(text=task["text"], interface="\n".join(f"- {line}" for line in task["interface"]))


def task_dir(instance, task_id):
    return os.path.join(put_paths.instance_dir(instance), "tasks", task_id)


def hidden_classes(task):
    return [os.path.splitext(os.path.basename(h))[0] for h in task["hidden"]]


def install_hidden(worktree, instance, task):
    """Copy the task's hidden tests into the worktree's test tree. -> the paths written."""
    src_root = os.path.join(task_dir(instance, task["id"]), "hidden")
    written = []
    for rel in task["hidden"]:
        dst = os.path.join(worktree, TEST_ROOT, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(src_root, *rel.split("/")), dst)
        written.append(dst)
    return written


def reference_patch(instance, task_id):
    with open(os.path.join(task_dir(instance, task_id), "reference.patch"), encoding="utf-8") as f:
        return f.read()


def new_unit_tests(patch_text):
    """-> simple names of the *UnitTest classes a patch adds under src/test."""
    names = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/src/test/") and line.endswith("UnitTest.java"):
            names.append(os.path.splitext(os.path.basename(line[6:]))[0])
    return names


def run_tests(worktree, instance, task, log_dir, own=()):
    """Install the hidden tests, compile, run hidden + pass-to-pass (+ the change's own unit tests).
    -> dict with the counts and reports."""
    install_hidden(worktree, instance, task)
    os.makedirs(log_dir, exist_ok=True)
    code, secs, tail = put_maven.compile_tests(worktree, log_path=os.path.join(log_dir, "compile.log"))
    hidden = hidden_classes(task)
    classes = hidden + list(task.get("pass_to_pass", ())) + [c for c in own if c not in hidden]
    if code != 0:
        return {"compiled": False, "compile_seconds": secs, "hidden": [0, 1], "pass_to_pass": [0, 1], "tail": tail[-1500:]}
    code, tsecs, tail = put_maven.test(worktree, classes, log_path=os.path.join(log_dir, "test.log"))
    rep = put_maven.reports(worktree, classes)
    return {"compiled": True, "compile_seconds": secs, "test_seconds": tsecs, "exit": code,
            "hidden": list(put_maven.pass_rate(rep, hidden)),
            "pass_to_pass": list(put_maven.pass_rate(rep, task.get("pass_to_pass", ()))),
            "own": list(put_maven.pass_rate(rep, own)) if own else [0, 0],
            "reports": {k: {kk: v[kk] for kk in ("tests", "failures", "errors", "skipped")} for k, v in rep.items()},
            "tail": "" if code == 0 else tail[-1500:]}


def validate(instance, source, base, work, only=None, rounds=2):
    rows = []
    for task in put_contract.tasks(instance):
        if only and task["id"] not in only:
            continue
        for r in range(rounds):
            for arm in ("base", "reference"):
                wt = os.path.join(work, f"{task['id']}-{arm}-{r}")
                put_worktree.create(source, base, wt)
                try:
                    own = ()
                    if arm == "reference":
                        patch = reference_patch(instance, task["id"])
                        put_worktree.apply(wt, patch)
                        own = new_unit_tests(patch)
                    res = run_tests(wt, instance, task, os.path.join(work, "logs", f"{task['id']}-{arm}-{r}"), own)
                finally:
                    put_worktree.remove(source, wt)
                h, p = res["hidden"], res["pass_to_pass"]
                if arm == "base":
                    ok = not res["compiled"] or h[0] < h[1]
                else:
                    o = res.get("own", [0, 0])
                    ok = res["compiled"] and h[0] == h[1] and h[1] > 0 and p[0] == p[1] and o[1] > 0 and o[0] == o[1]
                row = {"task": task["id"], "round": r, "arm": arm, "ok": ok, **{k: v for k, v in res.items() if k != "reports"}}
                rows.append(row)
                print(f"{task['id']:30s} r{r} {arm:9s} {'OK ' if ok else 'BAD'} compiled={res['compiled']} "
                      f"hidden={h[0]}/{h[1]} p2p={p[0]}/{p[1]} own={res.get('own', [0, 0])[0]}/{res.get('own', [0, 0])[1]}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "reference", "card"])
    ap.add_argument("--instance", default="backend-conventions")
    ap.add_argument("--source", default="C:/Users/Norbert/erdos-ws/backend")
    ap.add_argument("--work", default="C:/Users/Norbert/put-ws/validate")
    ap.add_argument("--task", action="append")
    ap.add_argument("--from", dest="from_wt")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    contract = put_contract.load(a.instance)
    if a.cmd == "card":
        for t in put_contract.tasks(a.instance):
            if not a.task or t["id"] in a.task:
                print(card(t), "\n" + "-" * 80)
        return 0
    if a.cmd == "reference":
        diff = put_worktree.capture(a.from_wt)
        with open(os.path.join(task_dir(a.instance, a.task[0]), "reference.patch"), "w", encoding="utf-8", newline="\n") as f:
            f.write(diff)
        print(f"reference.patch {len(diff)} bytes")
        return 0
    t0 = time.time()
    rows = validate(a.instance, a.source, contract["base_sha"], a.work, a.task, a.rounds)
    summary = {"instance": a.instance, "base_sha": contract["base_sha"], "rounds": a.rounds,
               "seconds": round(time.time() - t0, 1), "ok": all(r["ok"] for r in rows), "rows": rows}
    out = a.out or os.path.join(put_paths.instance_dir(a.instance), "tasks", "validation.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    print("ALL OK" if summary["ok"] else "PROBLEMS", f"{summary['seconds']} s -> {out}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
