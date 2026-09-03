# CodeMap API tier — the escalation adapter (docs/04 §7). ONE model-agnostic Claude
# Messages call; model choice is a config string, never a feature branch. Stdlib only
# (the app's zero-dependency law): raw HTTPS to the Messages API, SDK-compatible payload.
#
# Key resolution: ANTHROPIC_API_KEY in the process env, else .env at repo root
# (KEY=VALUE lines, gitignored). The key is never logged, echoed, or returned by any
# endpoint. No key -> tier disabled; the app runs local-only.
#
# Consent law: this module NEVER fires on its own. The server calls it only from
# POST /escalate, which the UI sends only when the user clicks the consent card.

import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# the model policy (docs/04 §7): one adapter, opinionated default, string-configurable.
# 'reasoning' = the Sonnet 4-line with CLASSIC extended thinking (budget_tokens) —
# still served on claude-sonnet-4-5; sonnet-5 (default) already thinks adaptively.
MODELS = {
    "default": "claude-sonnet-5",
    "budget": "claude-haiku-4-5-20251001",
    "deep": "claude-opus-5",
    "reasoning": "claude-sonnet-4-5",
}
# per-alias thinking config; budget_tokens must stay < max_tokens (API law)
THINKING = {
    "reasoning": {"type": "enabled", "budget_tokens": 6000},
}


def thinking_for(choice):
    """The thinking block an alias implies (None for adaptive-by-default models)."""
    return THINKING.get(choice)


def _read_env_file():
    path = os.path.join(ROOT, ".env")
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def api_key():
    # .env FIRST, ambient env second: the app's own config file is intentional; the
    # machine env proved to carry stale keys that shadowed a valid .env with 401s
    return _read_env_file().get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


def available():
    return bool(api_key())


def resolve_model(choice):
    """'default'|'budget'|'deep' aliases, or a verbatim model id string."""
    if not choice:
        return MODELS["default"]
    return MODELS.get(choice, choice)


def _call(body, timeout=120):
    """One Messages API request (tests monkeypatch THIS — never the network)."""
    key = api_key()
    if not key:
        raise RuntimeError("API tier disabled: no ANTHROPIC_API_KEY in .env or env")
    req = urllib.request.Request(API_URL, json.dumps(body).encode("utf-8"), {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def escalate(question, offer, model=None, max_tokens=1500, timeout=120):
    """One-shot escalation: trajectory as CONTEXT only (no tools). Kept as fallback."""
    traj = (offer or {}).get("context", {}).get("trajectory", [])
    reason = (offer or {}).get("reason", "")
    system = (
        "You are the escalation tier of CodeMap, a codebase navigator. The local "
        "navigator declined this question and provides its trajectory as context. "
        "Answer concisely and concretely about the checkItOut codebase; if the "
        "trajectory names files, treat them as the graph's pointers. If you cannot "
        "answer from the given context, say exactly what file content would be needed.")
    user = (f"QUESTION: {question}\n\nLocal navigator declined: {reason}\n"
            f"Navigator trajectory:\n" + "\n".join(f"  {t}" for t in traj[:8]))
    body = dict(model=resolve_model(model), max_tokens=max_tokens, system=system,
                messages=[dict(role="user", content=user)])
    th = thinking_for(model)
    if th:
        body["thinking"] = th
        body["max_tokens"] = max(max_tokens, th["budget_tokens"] + 2000)
    out = _call(body, timeout)
    answer = "".join(b.get("text", "") for b in out.get("content", [])
                     if b.get("type") == "text")
    rec = dict(answer=answer, model=out.get("model"),
               usage=out.get("usage", {}), tier="api")
    _log(question, offer, rec)
    return rec


# ---------------------------------------------------------------------------
# NAVIGATE mode — the Messages API as a first-class navigator (owner design
# 02.09 12:55): the model is TAUGHT the DSL and the graph structure in its
# system prompt and drives the SAME engine through one `cmdsl` tool. Same
# verbs, same digests, same open-book truth as the local 4B — intelligence
# stays in the graph; the API model is just a stronger tongue.
# ---------------------------------------------------------------------------

CMDSL_TOOL = dict(
    name="cmdsl",
    description=(
        "Execute ONE CodeMap DSL expression against the code graph and return its "
        "result digest. Verbs: map() overview+subsystem index; enter(sub_id) open a "
        "subsystem; find(term) fuzzy-search entity names; impact(entity[,depth]) who "
        "depends on it; flow(entity[,depth]) what it calls; seam(subA,subB) edges "
        "between two subsystems; cohort(entity) co-change hyperedge partners; "
        "spine(sub_id) the minimal reading path; health(coupling|hubs) global metrics; "
        "read(entity) the file's path pointer. Entity args must be EXACT names taken "
        "from earlier results (find/map first when unsure)."),
    input_schema=dict(type="object",
                      properties=dict(expression=dict(
                          type="string",
                          description='e.g. map() | find("stripe") | impact(UserRepository.java)')),
                      required=["expression"]))


def navigator_system(engine):
    """Teach the API model the graph structure + DSL protocol, grounded in THIS pack."""
    l1 = engine.map()
    return (
        "You are CodeMap's API navigator for the checkItOut codebase. A precomputed "
        "code graph holds the understanding; you NAVIGATE it with the `cmdsl` tool — "
        "you never guess file names or structure from memory.\n\n"
        "THE GRAPH: 3 levels. L1 = the subsystem index below (ids in [brackets]; GROUP "
        "rows contain child subsystems). L2 = one navigator per subsystem (enter(id) "
        "shows summary, entry points, spines, contracts, caveats). L3 = 1,415 entity "
        "files with dependency edges (17 typed relations), co-change hyperedge cohorts, "
        "and MacKay trophic heights (low = upstream/entry, high = deep dependency).\n\n"
        "PROTOCOL: (1) Start from the question's own words — find(term) for entity-ish "
        "questions, map()/enter() for subsystem-level ones. (2) THE SELECTION LAW: "
        "entity arguments must be EXACT names copied from a result you received, never "
        "invented. (3) Prefer the affordances results suggest ('next:'). (4) 2-5 tool "
        "calls usually suffice; stop when the evidence answers the question. (5) If the "
        "graph cannot answer (content-only, out of corpus, ambiguous), SAY SO plainly "
        "and name what file content would be needed — never fabricate.\n\n"
        "ANSWER STYLE: concise, concrete, cite file names and subsystem ids from your "
        "results; include any mermaid block a result provided when flow is asked.\n\n"
        f"L1 SUBSYSTEM INDEX:\n{l1.get('index', '')}\n\n"
        f"GLOBAL CAVEATS: {l1.get('caveats', '')}")


def navigate(engine, question, model=None, offer=None, max_steps=8, timeout=120):
    """Agentic escalation: Claude drives the engine via the cmdsl tool until it answers."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(ROOT, "app"))
    from dsl import execute as _execute

    msgs = [dict(role="user", content=f"QUESTION: {question}")]
    usage_in = usage_out = 0
    trajectory = []
    out = None
    th = thinking_for(model)  # extended thinking rides every request of the loop;
    # thinking blocks return in content and are echoed back verbatim (API law)
    for _ in range(max_steps):
        body = dict(model=resolve_model(model), max_tokens=1024,
                    system=navigator_system(engine), tools=[CMDSL_TOOL],
                    messages=msgs)
        if th:
            body["thinking"] = th
            body["max_tokens"] = th["budget_tokens"] + 2000
        out = _call(body, timeout)
        u = out.get("usage", {})
        usage_in += u.get("input_tokens", 0)
        usage_out += u.get("output_tokens", 0)
        if out.get("stop_reason") != "tool_use":
            break
        msgs.append(dict(role="assistant", content=out["content"]))
        results = []
        for block in out["content"]:
            if block.get("type") != "tool_use":
                continue
            expr = (block.get("input") or {}).get("expression", "")
            trajectory.append(expr)
            try:
                res = _execute(engine, expr)
                from serve_digest import digest_v2  # the ONE serve-contract renderer
                payload = digest_v2(res)
            except Exception as ex:
                payload = f"ERROR: {ex}"
            results.append(dict(type="tool_result", tool_use_id=block["id"],
                                content=payload[:4000]))
        msgs.append(dict(role="user", content=results))
    answer = "".join(b.get("text", "") for b in (out or {}).get("content", [])
                     if b.get("type") == "text")
    if out is not None and (out.get("stop_reason") != "end_turn" or not answer.strip()):
        # ANY non-answer ending (tool budget exhausted, max_tokens truncation mid-tool,
        # empty text) gets one forced SYNTHESIS turn from the gathered evidence. If the
        # last assistant turn was a truncated/unanswered tool_use, drop it — the API
        # rejects tool_use without a matching tool_result.
        if msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()
        msgs.append(dict(role="user", content="Tool budget exhausted. Answer the question "
                         "NOW from the evidence already gathered; name what is missing "
                         "if anything essential is."))
        body = dict(model=resolve_model(model), max_tokens=1500,
                    system=navigator_system(engine), tools=[CMDSL_TOOL],
                    tool_choice=dict(type="none"), messages=msgs)
        if th:
            body["thinking"] = th
            body["max_tokens"] = th["budget_tokens"] + 2000
        out = _call(body, timeout)
        u = out.get("usage", {})
        usage_in += u.get("input_tokens", 0)
        usage_out += u.get("output_tokens", 0)
        answer = "".join(b.get("text", "") for b in out.get("content", [])
                         if b.get("type") == "text")
    rec = dict(answer=answer, model=(out or {}).get("model"),
               usage=dict(input_tokens=usage_in, output_tokens=usage_out),
               trajectory=trajectory, steps=len(trajectory), tier="api-navigate")
    _log(question, offer, rec)
    return rec


def _log(question, offer, rec):
    """Provenance trail (answers only — never the key)."""
    path = os.path.join(ROOT, "eval", "q", "api_answers.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(question=question,
                                    offer_id=(offer or {}).get("offer_id"),
                                    reason_category=(offer or {}).get("reason_category"),
                                    **rec), ensure_ascii=False) + "\n")
    except OSError:
        pass
