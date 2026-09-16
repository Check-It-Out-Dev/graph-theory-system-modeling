"""GEPA on Modal: the same `run.py`, inside a CPU container, so the owner's box need not stay on.

    modal run eval/optimize/modal_gepa.py --dry-run                      # 5 navigator calls: proves the CLI bills the subscription from a container
    modal run eval/optimize/modal_gepa.py --train 8 --val 8 --max-metric-calls 60 --date 2026-09-17

Image: python 3.12 + real_ladybug + gepa + node 24 + @anthropic-ai/claude-code. The CodeMap tree is
mounted from the box at deploy time (pack excluded; the latest Release is fetched at start). The only
secret is `claude-oauth` (CLAUDE_CODE_OAUTH_TOKEN) — every model call inside is still `claude -p`.
Artifacts (run JSON, candidate, log) are written to the volume `codemap-train` under optimize/<date>/
and copied back to eval/optimize/runs/ by the local entrypoint. `promote.py` then runs on the box.
"""

import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
REMOTE_APP = "/root/codemap"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "git")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_24.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @anthropic-ai/claude-code@2.1.272",
    )
    .pip_install("real-ladybug==0.15.3", "gepa==0.1.4")
    .env({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "HOME": "/root"})
    .add_local_dir(R, remote_path=REMOTE_APP, ignore=["graph/pack/**", "**/__pycache__/**", "**/.pytest_cache/**", "dist/**",
                                                       "telemetry/events.jsonl", "search_index-*.jsonl", "eval/optimize/runs/**", ".git/**"])
)

app = modal.App("codemap-gepa", image=image)
vol = modal.Volume.from_name("codemap-train", create_if_missing=True)


@app.function(cpu=4, memory=8192, timeout=6 * 60 * 60, secrets=[modal.Secret.from_name("claude-oauth")], volumes={"/vol": vol})
def run_gepa(args: list[str], date: str, resume: bool = False) -> dict:
    import json
    import shutil
    import subprocess

    os.chdir(REMOTE_APP)
    subprocess.run([sys.executable, "tools/pack/fetch_pack.py", "--latest"], check=True)  # the pack is a Release, never in the image
    out_dir = f"/vol/optimize/{date}"
    # GEPA resumes silently from a run dir that exists: a second run under the same label would
    # load an exhausted checkpoint and do nothing (seen 2026-09-16). Fresh unless asked to resume.
    if os.path.isdir(f"{out_dir}/gepa") and not resume:
        shutil.rmtree(f"{out_dir}/gepa")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "eval/optimize/run.py", "--date", date, "--out", f"{out_dir}/{date}.json",
           "--candidate-out", f"{out_dir}/{date}.template.md", "--run-dir", f"{out_dir}/gepa"] + list(args)
    p = subprocess.run(cmd, capture_output=True, text=True)
    vol.commit()
    doc = None
    try:
        doc = json.load(open(f"{out_dir}/{date}.json", encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return {"returncode": p.returncode, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:], "doc": doc,
            "candidate": open(f"{out_dir}/{date}.template.md", encoding="utf-8").read() if os.path.exists(f"{out_dir}/{date}.template.md") else None}


@app.local_entrypoint()
def main(date: str = "", train: int = 8, val: int = 8, max_metric_calls: int = 60, dry_run: bool = False,
         model: str = "claude-sonnet-5", reflection_model: str = "claude-sonnet-5", seed: int = 0, resume: bool = False):
    import json
    import time

    date = date or time.strftime("%Y-%m-%d", time.gmtime())
    args = ["--train", str(train), "--val", str(val), "--max-metric-calls", str(max_metric_calls), "--model", model,
            "--reflection-model", reflection_model, "--seed", str(seed)] + (["--dry-run"] if dry_run else [])
    res = run_gepa.remote(args, date, resume)
    runs = os.path.join(HERE, "runs")
    os.makedirs(runs, exist_ok=True)
    if res.get("doc"):
        with open(os.path.join(runs, f"{date}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(res["doc"], f, indent=1, sort_keys=True)
    if res.get("candidate"):
        with open(os.path.join(runs, f"{date}.template.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(res["candidate"])
    print("returncode", res["returncode"])
    print(res["stdout"][-1500:])
    if res["returncode"] != 0:
        print(res["stderr"][-2500:])
