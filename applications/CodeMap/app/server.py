# CodeMap app server v0 — the docs/02 runtime shape in miniature: one process, browser UI,
# the SAME engine that the MCP transport and the small model will share. Stdlib only.
#
# Usage: PYTHONUTF8=1 python server.py [--port 7345]  ->  http://localhost:7345
# Endpoints: GET / (UI) · POST /step {"dsl": "..."} · POST /ask {"q": "..."} (protocol:
# cache first, then L1 injection — the mandated entry, enforced server-side).

import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# embedded-python law (installer E2E 03.09): a ._pth runtime pins sys.path and
# does NOT add the script's directory — pin it ourselves, harmless elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import api_tier
import big_tier
from dsl import execute, ParseError
from engine import Engine, DslError
from model_client import NavigatorModel

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = Engine()
MODEL = NavigatorModel()  # sidecar starts lazily on the first model-driven /ask
BIG = big_tier.BigNavigator()  # local big tier — same laziness, no consent needed

# EscalationOffer routing advice per trained reason category (docs/04 §7):
# needs-content-read escalates well; ambiguous should clarify, not spend; out-of-corpus
# is honest about likely waste; loop-health covers stall/invalid (the model not knowing
# that it is failing). Money never moves on the model's say-so — status stays PENDING_USER.
_CATEGORY_ADVICE = {
    "needs-content-read": "recommended: an API model with file access can finish what the graph pinned",
    "ambiguous": "clarify first — rephrasing is free, escalation is not",
    "out-of-corpus": "escalation likely wasted: the answer is not in this codebase",
    "multi-repo": "recommended: needs context beyond this pack",
    "loop-health": "recommended: the local navigator failed without knowing it",
}


def mint_offer(question, res):
    """Terminal pass()/stall/invalid -> the EscalationOffer contract (UI consent card)."""
    import hashlib
    reason = (res.get("text") or res["terminal"]).strip()
    cat = reason.split(":", 1)[0].strip().lower() if ":" in reason else "loop-health"
    if res["terminal"] in ("stall", "invalid"):
        cat = "loop-health"
    if cat not in _CATEGORY_ADVICE:
        cat = "out-of-corpus"
    oid = hashlib.sha256(f"{question}|{reason}".encode()).hexdigest()[:12]
    return dict(offer_id=oid, question=question, reason=reason, reason_category=cat,
                advice=_CATEGORY_ADVICE[cat],
                context=dict(trajectory=res.get("trajectory") or []),
                suggested_tier="api", suggested_model="claude-sonnet-5",
                status="PENDING_USER")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/status":
            # booleans only — the key itself never crosses this boundary
            return self._json(dict(
                entities=len(ENGINE.ents), navigators=len(ENGINE.l2),
                ladybug=bool(ENGINE.lb), local_model=MODEL.available(),
                big_model=BIG.available(), big_name=BIG.name(),
                api_tier=api_tier.available(), api_models=api_tier.MODELS,
                frozen="v1 (r2.2)"))
        if self.path in ("/", "/index.html"):
            b = open(os.path.join(HERE, "ui.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        try:
            if self.path == "/step":
                return self._json(execute(ENGINE, body.get("dsl", "")))
            if self.path == "/ask":
                q = body.get("q", "")
                hit = ENGINE.cache(q)  # the mandated first step, server-enforced
                if hit["kind"] == "cache_hit":
                    return self._json(dict(protocol="cache", **hit))
                if body.get("tier") == "big":
                    # explicit navigator choice — local, so no consent card
                    res = BIG.ask(ENGINE, q)
                    if res is None:
                        return self._json({"error": "big tier unavailable: no "
                                           "big gguf in bin/models/"}, 409)
                    return self._json(dict(protocol="big", cache=hit, **res))
                if MODEL.available() and not body.get("no_model"):
                    res = MODEL.ask(ENGINE, q)
                    if res is not None:
                        out = dict(protocol="model", cache=hit, **res)
                        if res["terminal"] in ("pass", "stall", "invalid"):
                            out["escalation_offer"] = mint_offer(q, res)
                        return self._json(out)
                return self._json(dict(protocol="descend", cache=hit, l1=ENGINE.map()))
            if self.path == "/escalate":
                # fires ONLY on the user's consent click in the UI — never automatically
                if body.get("target") == "big":
                    # retry-with-the-big-LOCAL-model: private, free, no key
                    res = BIG.ask(ENGINE, body.get("q", ""))
                    if res is None:
                        return self._json({"error": "big tier unavailable: no "
                                           "big gguf in bin/models/"}, 409)
                    return self._json(dict(protocol="big", **res))
                if not api_tier.available():
                    return self._json({"error": "API tier disabled: put "
                                       "ANTHROPIC_API_KEY in .env"}, 409)
                q = body.get("q", "")
                offer = body.get("offer") or {}
                try:
                    # navigate = the API model drives the engine itself (default);
                    # context = one-shot with trajectory as text (fallback)
                    if body.get("mode", "navigate") == "navigate":
                        rec = api_tier.navigate(ENGINE, q, model=body.get("model"),
                                                offer=offer)
                    else:
                        rec = api_tier.escalate(q, offer, model=body.get("model"))
                    return self._json(dict(protocol="api", **rec))
                except Exception as ex:
                    return self._json({"error": f"escalation failed: {ex}"}, 502)
            return self._json({"error": "unknown endpoint"}, 404)
        except (ParseError, DslError) as e:
            return self._json({"error": str(e)}, 422)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7345)
    port = ap.parse_args().port
    print(f"CodeMap v0: {len(ENGINE.ents)} entities, ladybug={'ON' if ENGINE.lb else 'OFF'} "
          f"-> http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()  # NOSONAR - loopback bind only; see sonar-project.properties


if __name__ == "__main__":
    main()
