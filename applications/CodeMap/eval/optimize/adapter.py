"""The GEPA adapter: a navigator prompt candidate is run against the local pack, scored by the oracle.

A candidate is `{"navigator_template": <template text>}`. To evaluate one example the adapter renders
the template over the pack (L1, L2, notes — the same builder the server uses), runs the navigator path
exactly as the VPS does (`claude -p` on the subscription, the engine as a loopback stdio MCP, the same
turn and effort limits), parses the answer's JSON block into pointers, and scores it with the execution
oracle from the judge: a gold entity among the first five pointers for a bank where-question, the
expected terminal for an off-distribution probe. No second model scores anything here; the judge's
rubric is for nights, the oracle is for optimisation loops that must not fool themselves.

The reflective dataset hands the reflector what it needs and nothing it should not see: the question,
the answer, the pointers, the engine steps taken, and a feedback line stating the gold that was missed
or the terminal that was expected. A candidate that violates `constraints.check` scores 0 on every
example with the violation as its feedback, so the proposer learns the contract instead of breaking it.
"""

import json
import os
import re
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(R, "app"), os.path.join(R, "tools", "prompt"), os.path.join(R, "eval", "judge"), R):
    sys.path.insert(0, p)

import claude_cli  # noqa: E402
import constraints  # noqa: E402
from remote.navigator import parse_answer  # noqa: E402

COMPONENT = "navigator_template"


def _read_trace(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


class _EvaluationBatch:
    """The shape GEPA reads (outputs, scores, trajectories) when gepa itself is not installed."""

    def __init__(self, outputs, scores, trajectories=None, objective_scores=None):
        self.outputs, self.scores, self.trajectories, self.objective_scores = outputs, scores, trajectories, objective_scores


class NavigatorAdapter:
    # gepa 0.1.4 reads these attributes (the Protocol calls them optional; the engine does not):
    # None means "use GEPA's default instruction proposal over the reflective dataset"
    propose_new_texts = None

    def __init__(self, pack_dir, notes_path, model="claude-sonnet-5", max_turns=10, effort="medium", timeout=240,
                 runner=None, log_path=None, oracle=None):
        self.pack_dir, self.notes_path = pack_dir, notes_path
        self.model, self.max_turns, self.effort, self.timeout = model, max_turns, effort, timeout
        self.runner, self.log_path = runner, log_path
        self.calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        self._rendered = {}
        if oracle is None:
            import judge
            oracle = judge.oracle
        self.oracle = oracle

    # ---------------------------------------------------------------- rendering
    def render(self, template_text):
        key = hash(template_text)
        if key in self._rendered and os.path.exists(self._rendered[key]):
            return self._rendered[key]
        import build_navigator
        fd, tpath = tempfile.mkstemp(prefix="nav-template-", suffix=".md")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(template_text)
        text, _ = build_navigator.build(tpath, self.pack_dir, self.notes_path)
        os.unlink(tpath)
        fd, ppath = tempfile.mkstemp(prefix="nav-prompt-", suffix=".md")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        self._rendered[key] = ppath
        return ppath

    def mcp_config(self, trace_file):
        env = {"CODEMAP_TRACE_FILE": trace_file, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "CODEMAP_PACK_DIR": self.pack_dir}
        return {"mcpServers": {"engine": {"command": sys.executable, "args": [os.path.join(R, "remote", "engine_mcp.py")], "env": env}}}

    # ---------------------------------------------------------------- one example
    def run_one(self, prompt_path, inst):
        trace = os.path.join(tempfile.gettempdir(), f"codemap-opt-trace-{os.getpid()}-{self.calls}.jsonl")
        self.calls += 1
        res = claude_cli.run(inst["q"], self.model, role="optimizer", system_file=prompt_path, mcp_config=self.mcp_config(trace),
                             allowed_tools=("mcp__engine__*",), max_turns=self.max_turns, effort=self.effort,
                             timeout=self.timeout, runner=self.runner)
        for k in self.usage:
            self.usage[k] += int((res.get("usage") or {}).get(k, 0) or 0)
        steps = [t.get("arg") for t in _read_trace(trace)]
        try:
            os.unlink(trace)
        except OSError:
            pass
        if res.get("is_error"):
            return {"answer": None, "pointers": [], "terminal": "error", "error": res.get("error"), "steps": steps, "rate_limited": res.get("rate_limited")}
        answer, names, terminal = parse_answer(res.get("text"))
        return {"answer": answer, "pointers": names, "terminal": terminal, "steps": steps}

    def score(self, out, inst):
        ev = {"pointers": [{"name": n} for n in out["pointers"]], "terminal": out["terminal"]}
        has, ok = self.oracle(ev, inst["kind"], inst["row"])
        if not has:
            return 0.0
        return 1.0 if ok else 0.0

    # ---------------------------------------------------------------- GEPA protocol
    def evaluate(self, batch, candidate, capture_traces=False):
        try:
            from gepa.core.adapter import EvaluationBatch
        except ImportError:  # the suites run without gepa (a dependency of eval/optimize only)
            EvaluationBatch = _EvaluationBatch
        template = candidate[COMPONENT]
        problems = constraints.check(template)
        outputs, scores, trajs = [], [], []
        if problems:
            for inst in batch:
                outputs.append({"answer": None, "pointers": [], "terminal": "refused", "steps": []})
                scores.append(0.0)
                trajs.append({"inst": inst, "out": outputs[-1], "feedback": "candidate refused before running: " + "; ".join(problems)})
            self._log({"event": "refused", "problems": problems, "n": len(batch)})
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs if capture_traces else None)
        prompt_path = self.render(template)
        for inst in batch:
            out = self.run_one(prompt_path, inst)
            s = self.score(out, inst)
            outputs.append(out)
            scores.append(s)
            trajs.append({"inst": inst, "out": out, "feedback": self.feedback(out, inst, s)})
            self._log({"event": "eval", "id": inst["id"], "kind": inst["kind"], "score": s, "terminal": out["terminal"],
                       "pointers": out["pointers"][:5], "steps": len(out["steps"]), "error": out.get("error")})
            if out.get("rate_limited"):
                raise RuntimeError("subscription rate limit reached — stop the run, resume later")
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs if capture_traces else None)

    def feedback(self, out, inst, score):
        row = inst["row"]
        if inst["kind"] == "probe":
            exp = row.get("expect")
            if score >= 1.0:
                return f"correct terminal '{exp}' for an off-distribution question."
            return f"wrong terminal: answered '{out['terminal']}', expected '{exp}' (kind {row.get('kind')})."
        import judge
        gold = sorted(judge.gold_entities(row))
        if score >= 1.0:
            return f"correct: a gold entity is among the first pointers ({', '.join(gold[:4])})."
        if out["terminal"] != "answer":
            return f"the navigator abstained or failed ({out['terminal']}); the graph holds the answer in {', '.join(gold[:4])}."
        return (f"missed: none of the first five pointers {out['pointers'][:5]} names a gold entity ({', '.join(gold[:4])}); "
                f"archetype {row.get('archetype')}; steps taken: {out['steps'][:6]}.")

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        rows = []
        for t in eval_batch.trajectories or []:
            inst, out = t["inst"], t["out"]
            rows.append({"Inputs": inst["q"],
                         "Generated Outputs": json.dumps({"answer": (out.get("answer") or "")[:600], "pointers": out.get("pointers", [])[:8],
                                                          "terminal": out.get("terminal"), "engine_steps": out.get("steps", [])[:8]}, ensure_ascii=False),
                         "Feedback": t["feedback"]})
        return {c: rows for c in components_to_update}

    def _log(self, rec):
        if not self.log_path:
            return
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def reflection_lm(model="claude-sonnet-5", runner=None, log_path=None, timeout=300):
    """GEPA's teacher: one tool-less `claude -p` per reflection, on the subscription (role reflector)."""
    usage = {"calls": 0, "output_tokens": 0}

    def call(prompt):
        if not isinstance(prompt, str):
            prompt = "\n\n".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in prompt)
        res = claude_cli.run(prompt, model, role="reflector", max_turns=1, timeout=timeout, runner=runner, tools=[])
        usage["calls"] += 1
        usage["output_tokens"] += int((res.get("usage") or {}).get("output_tokens", 0) or 0)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "reflect", "is_error": res.get("is_error"), "chars": len(res.get("text") or "")}) + "\n")
        if res.get("is_error"):
            raise RuntimeError(f"reflection failed: {res.get('error')}")
        return res.get("text") or ""
    call.usage = usage
    return call


def dataset(pack_dir, probes_path, n_train, n_val, seed=0, invalidated=()):
    """Bank where-questions the oracle can check + off-distribution probes, split train/val by seed."""
    import random
    import judge
    rows = [json.loads(l) for l in open(os.path.join(pack_dir, "mfq.jsonl"), encoding="utf-8") if l.strip()]
    insts = []
    for r in rows:
        if r["id"] in invalidated or r.get("gold_status") == "COVERAGE_GAP":
            continue
        if r.get("archetype") in judge.WHERE_ARCHETYPES and judge.gold_entities(r):
            insts.append({"id": r["id"], "kind": "bank", "q": r["q"], "row": r})
    if probes_path and os.path.exists(probes_path):
        for l in open(probes_path, encoding="utf-8"):
            if l.strip():
                p = json.loads(l)
                insts.append({"id": p["id"], "kind": "probe", "q": p["q"], "row": p})
    rnd = random.Random(seed)
    rnd.shuffle(insts)
    return insts[:n_train], insts[n_train:n_train + n_val]
