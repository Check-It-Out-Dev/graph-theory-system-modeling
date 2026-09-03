# The serve-contract digest — THE one rendering of engine results that models see
# mid-loop (training pairs, the local navigator runtime, and the API navigate tier all
# import THIS; two copies would be two instruments sharing no assumptions — L11).
# Caps are part of the contract: full L1 index (a sub id the model cannot see is a sub
# id it cannot legally select), 8 find hits, 420 chars + affordances otherwise.

import json


def digest_v2(result):
    k = result.get("kind")
    if k == "hits":
        rows = [f'{h["name"]} (sub {h["sub"]})' for h in (result.get("hits") or [])[:8]]
        return "hits: " + ("; ".join(rows) if rows else "NONE")
    if k == "l1":
        return (result.get("index") or "")[:6000]
    d = {x: v for x, v in result.items() if x not in ("rows", "affordances", "dsl", "done")}
    s = json.dumps(d, ensure_ascii=False)
    aff = result.get("affordances") or []
    return (s[:420] + ("..." if len(s) > 420 else "")) + (f" | next: {aff[:4]}" if aff else "")
