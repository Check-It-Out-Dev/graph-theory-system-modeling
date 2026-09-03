# CodeMap CPU deploy rung — the number that matters for the bundled product: the trained
# navigator as Q4_K_M GGUF under llama.cpp ON CPU, grammar-ON vs grammar-OFF, with real
# latency. Writes GEN files scorable by training/eval_harness.py (file: predictor) plus a
# latency profile. This is rung 1.5/2-CPU of the ladder (PIPELINE stage 6) and the start
# of the H-COMP grid (doc-01).
#
# Usage: PYTHONUTF8=1 python app/rung_cpu.py --gguf <path.gguf> [--grammar] [--cap-step 120]

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SERVER = os.path.join(ROOT, "bin", "llama", "b10155", "llama-server.exe")
GRAMMAR = os.path.join(ROOT, "graph", "pack", "codemap_vocab.gbnf")
MASTER = open(os.path.join(ROOT, "training", "master_prompt_v1.txt"), encoding="utf-8").read().strip()


def rows_for_eval(cap_step, split_file="test.jsonl"):
    rows = [json.loads(l) for l in
            open(os.path.join(ROOT, "training", "data", split_file), encoding="utf-8")]
    steps = sorted((r for r in rows if r["meta"]["kind"] == "step"),
                   key=lambda r: (r["meta"].get("src") or "", r["meta"]["rec"],
                                  r["messages"][1]["content"][:40]))
    k = max(1, len(steps) // cap_step)
    return steps[::k] + [r for r in rows if r["meta"]["kind"] != "step"]


def wait_health(port, tries=120):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def chat(port, user, max_tokens):
    body = json.dumps({"model": "codemap",
                       "messages": [{"role": "system", "content": MASTER},
                                    {"role": "user", "content": user}],
                       "temperature": 0, "max_tokens": max_tokens}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read().decode("utf-8"))
    return out["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--grammar", action="store_true", help="serve with the pack vocabulary GBNF")
    ap.add_argument("--port", type=int, default=7346)
    ap.add_argument("--cap-step", type=int, default=120)
    ap.add_argument("--split-file", default="test.jsonl",
                    help="which test snapshot to score (models pair with THEIR split)")
    ap.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    a = ap.parse_args()

    mode = "grammar" if a.grammar else "plain"
    cmd = [SERVER, "-m", a.gguf, "-c", "4096", "--port", str(a.port),
           "-t", str(a.threads), "--no-webui"]
    if a.grammar:
        cmd += ["--grammar-file", GRAMMAR]
    print("serving:", " ".join(cmd))
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert wait_health(a.port), "llama-server never became healthy"
        rows = rows_for_eval(a.cap_step, a.split_file)
        print(f"eval rows: {len(rows)} | mode: {mode}")
        out, lat_step, lat_ans = [], [], []
        for i, r in enumerate(rows):
            u = r["messages"][1]["content"]
            is_ans = r["meta"]["kind"] == "answer"
            t0 = time.perf_counter()
            gen = chat(a.port, u, 380 if is_ans else 64)
            dt = time.perf_counter() - t0
            (lat_ans if is_ans else lat_step).append(dt)
            out.append(dict(user=u, gen=gen, kind=r["meta"]["kind"], rec=r["meta"]["rec"]))
            if i % 25 == 0:
                print(f"[{i}/{len(rows)}] {dt:.2f}s {gen[:60]}")
        gen_path = os.path.join(ROOT, "training", "data", f"GEN_cpu-{mode}_test.jsonl")
        with open(gen_path, "w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        lat = {m: dict(n=len(v), mean=round(statistics.mean(v), 2),
                       p50=round(statistics.median(v), 2),
                       p95=round(sorted(v)[int(len(v) * 0.95) - 1], 2))
               for m, v in (("step", lat_step), ("answer", lat_ans)) if v}
        lat_path = os.path.join(ROOT, "training", "data", f"LAT_cpu-{mode}.json")
        json.dump(lat, open(lat_path, "w", encoding="utf-8"), indent=1)
        print("latency:", json.dumps(lat))
        print("saved:", gen_path)
    finally:
        srv.kill()


if __name__ == "__main__":
    main()
