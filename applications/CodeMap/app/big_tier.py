# CodeMap BIG local tier — an untrained big instruct model (Qwen3-Next-80B-A3B by
# default) as a first-class LOCAL navigator. No consent card: nothing leaves the
# machine. Where the API tier is "a stronger tongue" rented from the cloud, this tier
# is a stronger tongue that lives in bin/models/.
#
# What doc-06 measured is the design brief: prompt-only big models peak at 0.246
# route-success because canonical routes are learned, not promptable — but they reach
# TRUE answers by non-canonical routes (content F1 reversed the 30B/80B verdict). So
# this tier stops forcing the canonical surface and hands the model the graph's NATIVE
# language instead: the system prompt teaches HOW THE GRAPH WAS BUILT (so every field
# means something) and cypher(<stmt>) executes READ-ONLY against the pack's LadybugDB.
# The 13 CMDSL verbs stay available as cheap precompiled shapes; cypher() covers every
# shape the verbs cannot say. The 80B's other measured failure mode — stalling out the
# budget — gets the same forced-synthesis rescue the API tier has.
#
# Usage (smoke): PYTHONUTF8=1 python app/big_tier.py --q "how many files import
#   ResourceNotFoundException?"   [--port 7352]  (reuses a healthy server on the port)

import argparse
import json
import os
import re
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import sys
sys.path.insert(0, HERE)
from dsl import parse, ParseError, execute  # noqa: E402
from serve_digest import digest_v2  # noqa: E402
from rung_cpu import SERVER, wait_health  # noqa: E402

# quant preference: NL > XS (quality) > 30B (fallback); env pin wins outright
_GGUFS = [
    "Qwen_Qwen3-Next-80B-A3B-Instruct-IQ4_NL.gguf",
    "Qwen3-Next-80B-A3B-Instruct-IQ4_XS.gguf",
    "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
]
MAX_STEPS = 6
CYPHER_ROW_CAP = 25
CYPHER_CHAR_CAP = 900

# read-only law: the pack DB is an export, never a system of record — a model must
# not be able to write it even by accident. Token blocklist over one statement.
_FORBID = re.compile(
    r"\b(create|merge|delete|detach|set|remove|drop|alter|copy|load|call|install|"
    r"import|export|attach|detach|use|begin|commit|rollback|checkpoint)\b", re.I)
_HAS_LIMIT = re.compile(r"\blimit\s+\d+", re.I)
_CYPHER_RX = re.compile(r"^\s*cypher\s*\(\s*(.*?)\s*\)\s*;?\s*$", re.S)


def resolve_gguf():
    env = os.environ.get("CODEMAP_BIG_GGUF")
    if env:
        return env
    for name in _GGUFS:
        p = os.path.join(ROOT, "bin", "models", name)
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, "bin", "models", _GGUFS[0])  # absent -> tier off


# ---------------------------------------------------------------------------
# The system prompt — the construction story + the native language. Grounded in
# THIS pack at call time (L1 index injected), never memorized.
# ---------------------------------------------------------------------------

_STORY = """You are CodeMap's BIG local navigator for the checkItOut codebase (a
production Spring Boot + Angular marketplace: companies post campaigns,
influencers apply; Stripe billing, Fakturownia invoicing, Firebase auth, GDPR
consent enforcement, lifecycle crons). A precomputed knowledge graph holds the
understanding; you navigate it. Never answer from memory of similar codebases -
every claim must be grounded in a RESULT line you received in this session.

HOW THE GRAPH WAS BUILT - so you know what every field MEANS:
1. An indexer read all 1,415 source files of the two repos (backend
   checkItOut-be2 + frontend) and wrote one Entity row per FILE. Entities are
   files, not classes or functions.
2. A deterministic extractor mined 4,786 typed dependency edges. Counts by
   type: IMPORTS 2830, INJECTS 853, EXTENDS 317, PERFORMS 171, USES 163,
   TESTED_BY 113, MODIFIES 97, CALLS 91, ACCESSES 54, IMPLEMENTS 43,
   CONSTRAINS 13, VALIDATES 11, AFFECTS 9, TRIGGERS 6, APPLIES_IN 3,
   CONFIGURED_BY 2, INITIATES 2. TRIGGERS and TESTED_BY are under-extracted:
   event-flow and test-coverage results are shape signals, not censuses.
3. Community detection (Leiden over three fused embedding lenses) partitioned
   the corpus into measured subsystems; a curation pass reshaped them into the
   6-way tree in the L1 INDEX below (GROUP navigators, promoted LAYERs, two
   merged subsystems). Curation MOVED 455 files, so an entity's authoritative
   subsystem is coalesce(e.curated, e.subsystem) - curated when present, else
   the raw measured partition (kept for provenance).
4. A navigation pass computed per-file role layer (Actor / Process / Resource),
   MacKay trophic height (local_height: LOW = upstream entry point, HIGH = deep
   dependency), entry_point flags, and per-subsystem reading spines; it wrote
   an AI summary clue for every subsystem - that is what enter(id) returns.
5. Co-change mining produced 92 hyperedge cohorts (files that historically
   change together - cohort(entity) reads them); 104 frequent questions were
   pre-answered into a gold cache.

THE THREE LEVELS: L1 = the subsystem index (below). L2 = one navigator per
subsystem: enter(id) returns its summary, entry points, spines, contracts,
caveats. L3 = the 1,415 entities and their edges - where find/impact/flow/
seam/cohort and cypher() operate.

ACTIONS - respond with EXACTLY ONE action per turn, one physical line, no
prose, no markdown, no code fences, no semicolon:
  map()                    the L1 index (already below - rarely needed)
  enter(11)                open subsystem 11 (BARE integer - never enter(sub 11))
  find(Stripe)             ONE atomic term, entity-name lookup
  impact(Foo.java)         who depends on it (reverse edges), optional depth 2nd arg
  flow(Foo.java)           what it uses (forward edges), optional depth 2nd arg
  seam(3,11)               the edge bundle between two subsystems
  cohort(Foo.java)         co-change partners from the mined hyperedges
  spine(11)                the minimal reading path of subsystem 11
  health(coupling)         global metrics (coupling|hubs)
  read(Foo.java)           the file's path pointer (the graph pins WHERE)
  cypher(<one statement>)  NATIVE read-only query - see below
  answer("...")            terminate with the final answer (ONE line)
  pass("reason")           terminate honestly when the graph cannot answer

NATIVE QUERY: cypher(<statement>) runs read-only openCypher against the graph
database itself (LadybugDB, Kuzu-style dialect). Schema:
  Entity(name STRING, file_path STRING PK, entity_type STRING, subsystem INT64,
         curated INT64, layer STRING, local_height DOUBLE, entry_point BOOLEAN,
         spines STRING, line_count INT64)
  Dep(FROM Entity TO Entity, rel STRING)  -- ALL 17 edge types in ONE table
DIALECT LAWS (violating one wastes the step):
- Typed edges filter on the property: MATCH (a:Entity)-[r:Dep]->(b:Entity)
  WHERE r.rel = 'INJECTS' - NEVER [:INJECTS].
- Subsystem membership: coalesce(e.curated, e.subsystem) = <id>; ids are the
  [bracketed] numbers in the L1 INDEX.
- name is a label, not a key (two duplicate basenames exist); file_path is the
  primary key. No id() function exists.
- Substring search: WHERE lower(e.name) CONTAINS 'stripe'.
- ONE statement on ONE line; results are capped at 25 rows - aggregate with
  count(*), ORDER BY, LIMIT instead of listing everything. count(DISTINCT x)
  and RETURN key, count(*) grouping both work.
- CASE WHEN inside an aggregate (count/sum) returns WRONG numbers in this
  dialect - a conditional count MUST be its own query with a WHERE filter.
WHEN TO USE WHAT: the named verbs are precompiled and cheap - prefer them for
their exact shapes. Reach for cypher() when the question needs counting or
top-k, a SPECIFIC edge type, a multi-condition filter, or any shape no verb
covers. Worked examples:
  cypher(MATCH (a:Entity)-[r:Dep]->(b:Entity) WHERE r.rel = 'INJECTS' AND b.name = 'StripeService.java' RETURN a.name LIMIT 25)
  cypher(MATCH (e:Entity) WHERE coalesce(e.curated, e.subsystem) = 11 RETURN e.layer, count(*))
  cypher(MATCH (e:Entity) WHERE e.entry_point RETURN e.name, e.local_height ORDER BY e.local_height LIMIT 10)
  cypher(MATCH (a:Entity)-[r:Dep]->(b:Entity) WHERE lower(b.name) CONTAINS 'userrepository' RETURN r.rel, count(*))

SURFACE RULES a fine-tuned navigator knows and you must copy exactly:
- Entity arguments are EXACT names copied from the question or a RESULT line -
  never invented, never with parenthetical annotations.
- A RESULT line starting with ERROR means that exact expression failed - do
  not repeat it; change the verb, the argument, or the cypher.
- Never repeat an expression you already ran; its RESULT is already above.
- If two find() terms return nothing, try a camelCase or partial identifier
  (find(StepUp)) or a cypher CONTAINS query before passing.

HARD BUDGET: at most {max_steps} actions; each user turn is prefixed STEP
k/{max_steps}. Do not spend steps re-verifying what a RESULT already states -
by STEP {commit_step} you MUST commit: answer(...) grounded in the RESULTs you
already have, or pass(...) if the graph truly cannot answer. An answer from
partial evidence beats a stall. A reply that is prose commentary instead of an
action WASTES the whole step - the harness cannot execute a plan, only an
action.

ANSWER STYLE: concise and concrete; cite file names and [subsystem ids] from
your RESULTs; give the number when the question asks for one. The graph knows
WHERE things live and WHAT DEPENDS ON WHAT - it does not hold file content, so
questions needing code bodies, rationale, or history get pass("needs-content-
read: <which files>").
"""


def big_system(engine, max_steps=MAX_STEPS):
    l1 = engine.map()
    return (_STORY.format(max_steps=max_steps, commit_step=max(2, max_steps - 1))
            + "\nL1 SUBSYSTEM INDEX (ids in [brackets] are what enter() and the"
              " membership law take):\n" + (l1.get("index") or "")
            + "\n\nGLOBAL CAVEAT: subsystems are curated; curated and measured"
              " membership disagree BY DESIGN on 455 moved files - always"
              " coalesce(e.curated, e.subsystem)."
            + "\n\nFORMAT LAW (repeated last on purpose): your ENTIRE reply is"
              " exactly ONE action line - never prose, never a plan, never"
              " commentary about what a RESULT means. When you have evidence:"
              " answer(\"...\"). When the graph cannot answer: pass(\"...\").")


# ---------------------------------------------------------------------------
# cypher() — the guarded native path
# ---------------------------------------------------------------------------

def run_cypher(engine, stmt):
    """One read-only statement against the pack DB -> compact digest string."""
    if engine.lb is None:
        return ("ERROR: native path offline (Ladybug DB not loaded) - use the "
                "named verbs instead")
    q = stmt.strip().strip("'\"").rstrip(";").strip()
    if ";" in q:
        return "ERROR: ONE statement only (';' found)"
    m = _FORBID.search(q)
    if m:
        return f"ERROR: read-only tier - '{m.group(1)}' is not allowed"
    if not _HAS_LIMIT.search(q):
        q += f" LIMIT {CYPHER_ROW_CAP}"
    try:
        res = engine.lb.execute(q)
        cols = res.get_column_names()
        rows = []
        while res.has_next() and len(rows) < CYPHER_ROW_CAP:
            rows.append(res.get_next())
    except Exception as ex:
        return f"ERROR: {str(ex).splitlines()[0][:260]}"
    if not rows:
        return "cypher rows: NONE"
    head = "|".join(str(c) for c in cols)
    body = "; ".join("|".join(str(v) for v in r) for r in rows)
    out = f"cypher rows ({len(rows)}{'+' if len(rows) == CYPHER_ROW_CAP else ''}): {head} :: {body}"
    return out[:CYPHER_CHAR_CAP] + ("..." if len(out) > CYPHER_CHAR_CAP else "")


def clean_action(raw):
    """Liberal-decoding tolerance (bench-proven): fences, labels, bullets ->
    the one action line; multi-line answer()/pass() bodies collapse to one."""
    txt = re.sub(r"(?:^```[a-z]*\n?)|(?:```$)", "", (raw or "").strip(), flags=re.M).strip()
    lines = txt.splitlines()
    for i, line in enumerate(lines):
        line = line.strip().lstrip("-*> ").strip()
        line = re.sub(r"^(CMDSL|DSL|Action|Step \d+)\s*[:>]\s*", "", line, flags=re.I)
        if re.match(r"^(answer|pass)\(", line) and line.count('"') % 2 == 1:
            line = (line + " " + " ".join(x.strip() for x in lines[i + 1:])).strip()
        if re.match(r"^cypher\s*\(", line, re.I):
            # a cypher body may spill across lines inside fences - rejoin then cut
            joined = " ".join([line] + [x.strip() for x in lines[i + 1:]])
            depth = 0
            for j, ch in enumerate(joined):
                depth += ch == "("
                depth -= ch == ")"
                if depth == 0 and ch == ")":
                    return joined[: j + 1]
            return joined
        if re.match(r"^[a-z_]+\(", line):
            return re.sub(r"\)\s*;\s*$", ")", line)  # Cypher-primed ';' strip
    return lines[0].strip() if lines else ""


# ---------------------------------------------------------------------------
# The loop — same accumulating-transcript contract as loop_runner, plus the
# cypher branch and (product mode) lenient invalids + forced synthesis.
# ---------------------------------------------------------------------------

def chat(port, system, user, max_tokens, timeout=900):
    body = json.dumps({"model": "codemap-big", "temperature": 0,
                       "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",  # NOSONAR - loopback URL, operator-chosen port; see sonar-project.properties
                                 body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode("utf-8"))
    return out["choices"][0]["message"]["content"].strip()


def run_big_loop(port, engine, question, max_steps=MAX_STEPS, strict=False,
                 system=None):
    """Returns (trajectory, terminal, answer_text, results, backtracks).
    strict=True is bench mode: invalid ends the loop, no synthesis rescue.
    system overrides the prompt (bench ablation rungs); default = big_system."""
    system = system or big_system(engine, max_steps)
    user = f"QUESTION: {question}\n(cache: miss)"
    traj, results, seen, backtracks = [], [], set(), 0
    for step in range(max_steps):
        raw = chat(port, system, user + f"\nSTEP {step + 1}/{max_steps}:", 560)
        gen = clean_action(raw)
        traj.append(gen)
        cy = _CYPHER_RX.match(gen)
        if cy:
            if gen in seen:
                backtracks += 1
                user += (f"\nNOTE: you already ran {gen}; its RESULT is above. "
                         "Emit a DIFFERENT action, or answer()/pass() now.")
                continue
            seen.add(gen)
            dig = run_cypher(engine, cy.group(1))
            results.append(dict(dsl=gen, digest=dig))
            user += f"\nRESULT of {gen[:160]}: {dig}"
            continue
        try:
            verb, args = parse(gen)
        except ParseError:
            if strict:
                return traj, "invalid", None, results, backtracks
            user += (f"\nNOTE: not a valid action: {gen[:120]} - emit exactly "
                     "one action line from the ACTIONS list.")
            continue
        if verb == "answer":
            return traj, "answer", args[0], results, backtracks
        if verb == "pass":
            return traj, "pass", args[0], results, backtracks
        if gen in seen:
            backtracks += 1
            user += (f"\nNOTE: you already ran {gen}; its RESULT is above. "
                     "Emit a DIFFERENT action, or answer()/pass() now.")
            continue
        seen.add(gen)
        try:
            res = execute(engine, gen)
            dig = digest_v2(res)
        except Exception as ex:
            dig = f"ERROR: {type(ex).__name__}: {str(ex)[:200]}"
        results.append(dict(dsl=gen, digest=dig))
        user += f"\nRESULT of {gen}: {dig}"
    if strict:
        return traj, "stall", None, results, backtracks
    # forced synthesis — the API tier's rescue, ported: budget gone, commit now.
    raw = chat(port, system, user + "\nBUDGET EXHAUSTED. Emit answer(\"...\") NOW "
               "from the RESULT lines above (one line), or pass(\"reason\").", 700)
    gen = clean_action(raw)
    traj.append(gen)
    try:
        verb, args = parse(gen)
        if verb in ("answer", "pass"):
            return traj, verb, args[0], results, backtracks
    except ParseError:
        pass
    if raw.strip():  # prose fallback: better an ungrammatical answer than none
        return traj, "answer", raw.strip().splitlines()[0][:600], results, backtracks
    return traj, "stall", None, results, backtracks


# ---------------------------------------------------------------------------
# Sidecar owner — mirrors NavigatorModel so the server treats tiers uniformly.
# ---------------------------------------------------------------------------

class BigNavigator:
    """Lazy llama-server sidecar for the big gguf. Reuses an already-healthy
    server on the port (dev flow: boot once, iterate many)."""

    def __init__(self, gguf=None, port=None):
        self.gguf = gguf or resolve_gguf()
        self.port = int(port or os.environ.get("CODEMAP_BIG_PORT") or 7352)
        self.proc = None
        self._external = False

    def name(self):
        return os.path.splitext(os.path.basename(self.gguf))[0]

    def available(self):
        if os.path.exists(self.gguf) and os.path.exists(SERVER):
            return True
        return self._probe()

    def _probe(self):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=0.4) as r:
                self._external = r.status == 200
        except Exception:
            self._external = False
        return self._external

    def _ensure(self):
        if self._external or self._probe():
            return True
        if self.proc and self.proc.poll() is None:
            return wait_health(self.port, tries=5)
        if not (os.path.exists(self.gguf) and os.path.exists(SERVER)):
            return False
        self.proc = subprocess.Popen(
            [SERVER, "-m", self.gguf, "-c", "12288", "--port", str(self.port),
             "-t", str(max(2, (os.cpu_count() or 8) - 2)), "--no-webui"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # an 80B gguf streams from disk for minutes on a cold cache — be patient
        return wait_health(self.port, tries=420)

    def ask(self, engine, question, max_steps=MAX_STEPS):
        if not self._ensure():
            return None
        t0 = time.perf_counter()
        traj, terminal, text, results, backtracks = run_big_loop(
            self.port, engine, question, max_steps)
        return dict(trajectory=traj, terminal=terminal, text=text,
                    steps=len(traj), backtracks=backtracks, results=results,
                    seconds=round(time.perf_counter() - t0, 1),
                    model=self.name(), tier="local-big")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    a = ap.parse_args()
    from engine import Engine
    engine = Engine()  # use_ladybug=True: the native path is the point
    nav = BigNavigator(port=a.port)
    print(f"model: {nav.name()}  port: {nav.port}  ladybug: {'ON' if engine.lb else 'OFF'}")
    res = nav.ask(engine, a.q, a.steps)
    if res is None:
        print("big tier unavailable (no gguf / no server)")
        return 1
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
