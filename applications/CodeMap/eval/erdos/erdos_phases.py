"""Split one `claude -p --output-format stream-json` run into its solve and verify phases.

The task card asks every agent to write its answer, then the line `=== ANSWER COMPLETE ===`, then
to check the claims its plan rests on and write `=== VERIFIED ===` with corrections. The owner's
rule for the comparison: when verification is needed, tokens are counted separately before and
after it, so the cost of solving is not mixed with the cost of checking.

Accounting, per phase:
- API calls: an assistant message is one call. Stream-json sends one `assistant` event per content
  block (thinking, text, tool use), all with the message's id; each id is counted once. The usage on
  those events freezes `output_tokens` at the start of the message, so the runner also asks for
  partial messages (`--include-partial-messages`) and keeps each message's `message_start` and
  `message_delta`: the delta carries the final usage, and the per-call sums then equal the run's
  result totals (checked on a recorded stream in the tests).
- tokens: `input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens`,
  their plain sum, and a price-weighted sum (cache reads at 0.1, cache writes at 2 of an input token
  because Claude Code writes the one-hour cache, output at 5) so a cache-heavy run is not read as expensive by raw count alone.
- tool calls by tool name, and the bytes of tool results that entered the context.
- wall seconds, from the receive times the runner recorded per event.

The solve phase ends with the call whose text holds the marker; a run with no marker is all solve
and is flagged `marker: false`.
"""

import json
from collections import Counter

SOLVED = "=== ANSWER COMPLETE ==="
VERIFIED = "=== VERIFIED ==="
KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
WEIGHTS = {"input_tokens": 1.0, "cache_creation_input_tokens": 2.0, "cache_read_input_tokens": 0.1, "output_tokens": 5.0}


def load(path):
    """-> [(t, event)] from a runner events file (one {"t": seconds, "event": {...}} per line)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out.append((float(row.get("t", 0.0)), row.get("event") or {}))
    return out


def _result_bytes(block):
    c = block.get("content")
    if isinstance(c, list):
        return sum(len(json.dumps(x, ensure_ascii=False)) if not isinstance(x, dict) or x.get("type") != "text"
                   else len(x.get("text", "")) for x in c)
    return len(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))


def calls(events):
    """-> ordered API calls [{id, t_first, t_last, usage, text, tools, result_bytes}]; tool results are
    attributed to the call that requested them (the last call before the result arrived)."""
    out, index = [], {}
    current = None

    def call(mid, t):
        if mid not in index:
            index[mid] = len(out)
            out.append({"id": mid, "t_first": t, "t_last": t, "usage": {k: 0 for k in KEYS}, "text": "", "tools": [],
                        "result_bytes": 0})
        c = out[index[mid]]
        c["t_last"] = max(c["t_last"], t)
        return c

    def take_usage(c, usage):
        for k in KEYS:
            c["usage"][k] = max(c["usage"][k], int((usage or {}).get(k, 0) or 0))

    for t, e in events:
        kind = e.get("type")
        if kind == "stream_event":
            raw = e.get("event") or {}
            if raw.get("type") == "message_start":
                current = (raw.get("message") or {}).get("id")
                if current:
                    take_usage(call(current, t), (raw.get("message") or {}).get("usage"))
            elif raw.get("type") == "message_delta" and current:
                take_usage(call(current, t), raw.get("usage"))
            continue
        if kind == "assistant":
            m = e.get("message") or {}
            c = call(m.get("id") or f"anon-{len(out)}", t)
            take_usage(c, m.get("usage"))
            for block in m.get("content") or []:
                if block.get("type") == "text":
                    c["text"] += block.get("text", "")
                elif block.get("type") == "tool_use":
                    c["tools"].append(block.get("name") or "?")
        elif kind == "user" and out:
            for block in (e.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    out[-1]["result_bytes"] += _result_bytes(block)
    return out


def _aggregate(part, t_start, t_end):
    tokens = {k: sum(c["usage"][k] for c in part) for k in KEYS}
    tools = Counter(name for c in part for name in c["tools"])
    return {"calls": len(part), "tokens": tokens, "tokens_sum": sum(tokens.values()),
            "tokens_weighted": round(sum(tokens[k] * WEIGHTS[k] for k in KEYS), 1),
            "tool_calls": sum(tools.values()), "tools": dict(sorted(tools.items())),
            "graph_tool_calls": sum(n for name, n in tools.items() if name.startswith(("mcp__engine__", "mcp__graph__"))),
            "result_bytes": sum(c["result_bytes"] for c in part),
            "seconds": round(max(0.0, t_end - t_start), 1) if part else 0.0}


def split(events):
    """-> {marker, verified_marker, solve, verify, total, result, answer, corrections}."""
    cs = calls(events)
    k = next((i for i, c in enumerate(cs) if SOLVED in c["text"]), None)
    solve, verify = (cs[:k + 1], cs[k + 1:]) if k is not None else (cs, [])
    t0 = events[0][0] if events else 0.0
    t_split = cs[k]["t_last"] if k is not None else (events[-1][0] if events else 0.0)
    t_end = events[-1][0] if events else 0.0
    result = next((e for _, e in reversed(events) if e.get("type") == "result"), {}) or {}
    texts = [c["text"] for c in cs]
    if k is not None:
        head = cs[k]["text"].split(SOLVED)[0].strip()
        answer = head if len(head) >= 400 else "\n\n".join(t for t in texts[:k] + [head] if t.strip()).strip()
    else:
        answer = "\n\n".join(t for t in texts if t.strip()).strip()
    tail = "\n".join(texts[k:] if k is not None else [])
    corrections = tail.split(VERIFIED, 1)[1].strip() if VERIFIED in tail else None
    return {"marker": k is not None, "verified_marker": corrections is not None,
            "solve": _aggregate(solve, t0, t_split), "verify": _aggregate(verify, t_split, t_end),
            "total": _aggregate(cs, t0, t_end),
            "result": {"subtype": result.get("subtype"), "is_error": result.get("is_error"), "num_turns": result.get("num_turns"),
                       "duration_ms": result.get("duration_ms"), "total_cost_usd": result.get("total_cost_usd"),
                       "usage": {k2: int((result.get("usage") or {}).get(k2, 0) or 0) for k2 in KEYS}},
            "answer": answer, "corrections": corrections}
