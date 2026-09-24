"""Evidence: bring a campaign's Actions artifact into the repository and record it in RUNS.md.

    PYTHONUTF8=1 python eval/put/put_evidence.py pull --run-id <Actions run id> --label <label>

The job on norbert-box uploads the run directories as the artifact `put-<label>`. This command downloads it, copies
the label's directories and summary into `eval/put/runs/` (the stream files stay out of git by .gitignore; their
sha256 sit in each run's meta.json), and appends one row to RUNS.md: the run's URL, the artifact's id and digest,
the label, the prompt, the runs and the score. A number enters the README only when its campaign has a row here and
its artifacts are committed (law 8).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import put_paths

REPO = "Check-It-Out-Dev/checkitout-backend"
RUNS_MD = os.path.join(put_paths.HERE, "RUNS.md")
HEADER = ("# Campaign runs\n\nEvery prompt-under-test campaign that produced a committed number: its Actions run on the "
          "self-hosted runner, the artifact it uploaded (id and sha256 digest, 90-day retention; the full streams are "
          "also in the release asset named in the last column when one exists), and what it measured.\n\n"
          "| date | label | mode | Actions run | artifact (id, digest) | prompt | runs | score | release asset |\n"
          "|---|---|---|---|---|---|---|---|---|\n")


def gh_json(args):
    out = subprocess.run(["gh"] + args, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return json.loads(out)


def pull(run_id, label):
    arts = gh_json(["api", f"repos/{REPO}/actions/runs/{run_id}/artifacts"])["artifacts"]
    art = next(a for a in arts if a["name"] == f"put-{label}")
    tmp = tempfile.mkdtemp(prefix="put-art-")
    subprocess.run(["gh", "run", "download", str(run_id), "-R", REPO, "-n", art["name"], "-D", tmp], check=True)
    src_dir, src_json = os.path.join(tmp, label), os.path.join(tmp, f"{label}.json")
    shutil.copytree(src_dir, os.path.join(put_paths.RUNS, label), dirs_exist_ok=True)
    if os.path.exists(src_json):
        shutil.copyfile(src_json, os.path.join(put_paths.RUNS, f"{label}.json"))
    with open(src_json, encoding="utf-8") as f:
        summary = json.load(f)
    run = gh_json(["run", "view", str(run_id), "-R", REPO, "--json", "url,createdAt"])
    row = (f"| {run['createdAt'][:10]} | {label} | {summary.get('mode')} | [{run_id}]({run['url']}) | "
           f"{art['id']}, `{(art.get('digest') or '').replace('sha256:', '')[:16]}` | `{summary.get('prompt_version')}` | "
           f"{summary.get('runs')} | {summary.get('score')} | |\n")
    if not os.path.exists(RUNS_MD):
        with open(RUNS_MD, "w", encoding="utf-8", newline="\n") as f:
            f.write(HEADER)
    with open(RUNS_MD, encoding="utf-8") as f:
        existing = f.read()
    if f"| {label} |" not in existing:
        with open(RUNS_MD, "a", encoding="utf-8", newline="\n") as f:
            f.write(row)
    shutil.rmtree(tmp, ignore_errors=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args(argv)
    print(pull(a.run_id, a.label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
