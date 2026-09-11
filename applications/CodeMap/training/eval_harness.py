# CodeMap evaluation harness v2 (PIPELINE.md) — the automated gate set for ANY predictor:
# step_exact, exec_fp (engine execution fingerprints), ans_cos (Modal Qwen3-Embedding-8B),
# rr_equiv (Modal reranker gold-equivalence), topo_arch (archetype kNN consistency in
# embedding space), risk_cov (pass on unanswerable vs answerable).
#
# A predictor is anything with .step(user_text)->dsl and .answer(user_text)->text.
# ScriptedPredictor (rung 0.5) replays references — proves the harness end-to-end.
#
# Usage: PYTHONUTF8=1 python eval_harness.py --split test [--predictor scripted] [--limit N]

import argparse, hashlib, json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
from engine import Engine  # noqa: E402
from dsl import execute, parse, ParseError  # noqa: E402

EMBED_URL = "https://ramzesx--v3-code-embeddings-serve.modal.run"
RERANK_URLS = ["https://ramzesx--v3-code-reranker-serve.modal.run"]


def _post(url, obj, timeout=240):
    req = urllib.request.Request(url, json.dumps(obj).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def embed(texts):
    out = _post(EMBED_URL, {"texts": texts})
    return out["embeddings"] if isinstance(out, dict) else out


def rerank(query, docs):
    last = None
    for url in RERANK_URLS:
        for payload in ({"query": query, "documents": docs},
                        {"pairs": [[query, d] for d in docs]}):
            try:
                out = _post(url, payload)
                if isinstance(out, dict):
                    for k in ("scores", "results", "relevance_scores"):
                        if k in out:
                            v = out[k]
                            return [x["score"] if isinstance(x, dict) else x for x in v]
                if isinstance(out, list):
                    return [x["score"] if isinstance(x, dict) else x for x in out]
            except Exception as ex:
                last = ex
    raise RuntimeError(f"reranker unusable: {last}")


def cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    return num / (da * db + 1e-12)


def canon_dsl(s):
    try:
        v, args = parse(s)
        return f"{v}({','.join(a.strip() for a in args)})"
    except ParseError:
        return f"INVALID:{s[:60]}"


def result_fp(engine, dsl_expr):
    try:
        r = execute(engine, dsl_expr)
        r = {k: v for k, v in r.items() if k not in ("affordances", "dsl")}
        return hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False,
                                         default=str).encode()).hexdigest()[:16]
    except Exception as ex:
        return f"ERR:{type(ex).__name__}"


class FilePredictor:
    """Rung-1/2: replays a model's generations from modal_eval output (keyed by user text)."""
    def __init__(self, path):
        self.ref = {}
        for l in open(path, encoding="utf-8"):  # NOSONAR - operator's own path; see sonar-project.properties
            r = json.loads(l)
            self.ref[r["user"]] = r["gen"]

    def step(self, user):
        g = self.ref.get(user, 'pass("no generation for this input")')
        return g.splitlines()[0].strip() if g else g

    def answer(self, user):
        g = self.ref.get(user, 'answer("no generation")')
        m = re.match(r'answer\("(.*)"\)\s*$', g, re.S)
        return m.group(1) if m else g


class ScriptedPredictor:
    """Rung 0.5: replays the reference outputs — the harness-proof ceiling."""
    def __init__(self, rows):
        self.ref = {r["messages"][1]["content"]: r["messages"][2]["content"] for r in rows}

    def step(self, user):
        return self.ref.get(user, 'pass("scripted: unseen input")')

    def answer(self, user):
        out = self.ref.get(user, 'answer("scripted: unseen")')
        m = re.match(r'answer\("(.*)"\)\s*$', out, re.S)
        return m.group(1) if m else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--predictor", default="scripted")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-modal", action="store_true", help="skip embedding/reranker gates")
    a = ap.parse_args()

    rows = [json.loads(l) for l in
            open(os.path.join(HERE, "data", f"{a.split}.jsonl"), encoding="utf-8")]  # NOSONAR - operator's own path; see sonar-project.properties
    if a.limit:
        rows = rows[: a.limit]
    if a.predictor.startswith("file:"):
        pred = FilePredictor(a.predictor[5:])
        # score ONLY rows the generation file covers (modal_eval subsamples the step rows);
        # a missing generation is "not evaluated", never a synthetic failure
        n0 = len(rows)
        rows = [r for r in rows if r["messages"][1]["content"] in pred.ref]
        print(f"file predictor covers {len(rows)}/{n0} rows of split '{a.split}'")
    else:
        pred = ScriptedPredictor(rows)  # rung 0.5 ceiling
    eng = Engine(use_ladybug=False)

    step_rows = [r for r in rows if r["meta"]["kind"] in ("step", "loop")]
    ans_rows = [r for r in rows if r["meta"]["kind"] == "answer"]
    pass_rows = [r for r in rows if r["meta"]["kind"] == "pass"]

    # --- DSL-level ---
    se = ef = 0
    for r in step_rows:
        got, ref = pred.step(r["messages"][1]["content"]), r["messages"][2]["content"]
        if canon_dsl(got) == canon_dsl(ref):
            se += 1
        if result_fp(eng, got) == result_fp(eng, ref):
            ef += 1
    # --- abstention ---
    good_pass = sum(1 for r in pass_rows
                    if pred.step(r["messages"][1]["content"]).startswith("pass("))
    false_pass = sum(1 for r in step_rows
                     if pred.step(r["messages"][1]["content"]).startswith("pass("))
    metrics = dict(
        n=dict(step=len(step_rows), answer=len(ans_rows), abstain=len(pass_rows)),
        step_exact=round(se / max(1, len(step_rows)), 4),
        exec_fp=round(ef / max(1, len(step_rows)), 4),
        risk_cov=dict(pass_on_unanswerable=round(good_pass / max(1, len(pass_rows)), 4),
                      false_pass_on_answerable=round(false_pass / max(1, len(step_rows)), 4)),
    )

    # --- answer-level (Modal judges) ---
    if ans_rows and not a.no_modal:
        golds, preds, archetypes, qs = [], [], [], []
        for r in ans_rows:
            u = r["messages"][1]["content"]
            golds.append(re.match(r'answer\("(.*)"\)\s*$', r["messages"][2]["content"],
                                  re.S).group(1))
            preds.append(pred.answer(u))
            qs.append(u.split("\n")[0].replace("QUESTION: ", ""))
            # archetype key: templated null-answers form a CLASS (any null-impact near
            # another null-impact is correct retrieval); gold answers remain self-keyed
            src = r["meta"].get("src") or ""
            archetypes.append(src if src.startswith("null-") else r["meta"]["rec"])
        ge = embed(golds)
        pe = embed(preds)
        cs = [cos(x, y) for x, y in zip(ge, pe)]
        metrics["ans_cos"] = dict(mean=round(sum(cs) / len(cs), 4),
                                  frac_ge_085=round(sum(1 for c in cs if c >= 0.85) / len(cs), 4))
        # topo_arch: each predicted answer's nearest GOLD must share its ARCHETYPE key —
        # self-retrieval for unique golds, class-retrieval for templated null answers
        hits = 0
        for i, p in enumerate(pe):
            best = max(range(len(ge)), key=lambda j: cos(p, ge[j]))
            hits += (archetypes[best] == archetypes[i])
        metrics["topo_arch"] = round(hits / len(pe), 4)
        # rr_equiv on a bounded sample
        k = min(12, len(qs))
        eq = 0
        for i in range(k):
            s_pred, s_gold = rerank(qs[i], [preds[i], golds[i]])[:2]
            eq += (s_pred >= 0.90 * s_gold)
        metrics["rr_equiv"] = dict(sample=k, frac=round(eq / max(1, k), 4))

    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    tag = re.sub(r"[^A-Za-z0-9_-]+", "_", a.predictor)[:40]
    out = os.path.join(HERE, "data", f"EVAL_{tag}_{a.split}.json")
    json.dump(metrics, open(out, "w", encoding="utf-8"), indent=1)  # NOSONAR - operator's own path; see sonar-project.properties


if __name__ == "__main__":
    main()
