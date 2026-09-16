"""Ask the navigator a slice of the bank and the probes as a system user with the FAQ skipped, so the
judge has answers with an oracle to calibrate against. The server's events file is the artifact;
this script only drives it.

    python eval/judge/bank_pass.py --url http://127.0.0.1:7391 --token t0k --user judge --n 24 --probes 6 [--seed 1]
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))


def ask(url, token, user, q, no_faq=True, timeout=400):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "codemap_ask", "arguments": {"q": q, "no_faq": no_faq}}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/mcp", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}",
                                          "X-CodeMap-User": user})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # NOSONAR - our own server; see sonar-project.properties
        res = json.load(r)
    out = json.loads(res["result"]["content"][0]["text"])
    return out, bool(res["result"].get("isError"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("CODEMAP_URL", "http://127.0.0.1:7345"))
    ap.add_argument("--token", default=os.environ.get("CODEMAP_TOKEN", ""))
    ap.add_argument("--user", default="judge")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--probes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    bank = [json.loads(l) for l in open(os.path.join(R, "eval", "q", "mfq_all.jsonl"), encoding="utf-8") if l.strip()]
    probes = [json.loads(l) for l in open(os.path.join(R, "eval", "q", "probes_offdist.jsonl"), encoding="utf-8") if l.strip()]
    rnd = random.Random(a.seed)
    executed = [r for r in bank if r.get("gold_status") == "EXECUTED"]
    picks = rnd.sample(executed, min(a.n, len(executed)))
    abst = [p for p in probes if p["expect"] == "abstain"]
    picks += rnd.sample(abst, min(a.probes, len(abst)))
    t0 = time.time()
    for i, row in enumerate(picks, 1):
        try:
            out, err = ask(a.url, a.token, a.user, row["q"])
            print(f"{i:2}/{len(picks)} {row['id']:5} {out.get('tier','?'):10} {out.get('terminal','?'):8} "
                  f"steps={out.get('steps')} credits={out.get('credits')} {'ERR ' + str(out.get('error'))[:60] if err else ''}")
            if err and out.get("terminal") == "budget_exhausted":
                print("budget exhausted; stopping")
                break
        except Exception as ex:
            print(f"{i:2}/{len(picks)} {row['id']:5} failed: {type(ex).__name__}: {str(ex)[:120]}")
    print(f"done in {int(time.time() - t0)} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
