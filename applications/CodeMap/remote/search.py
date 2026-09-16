"""Semantic search over the pack: Qwen3-Embedding-8B for recall, Qwen3-Reranker-8B for precision
(both on Modal, the only non-Claude models in the loop). The pack ships no embeddings (they are
authoring-only by SCHEMA §4), so the index is built once per pack version from a cheap socket
per entity (name, type, package words, subsystem name and responsibilities) and cached beside
the telemetry. Pure-python cosine over 1,415 x 4,096 floats takes well under a second.

Modal answers a request that exceeds ~150 s with 303 + a polling URL (cold starts do); the
client follows it. Search costs a flat `search_call` credit plus the query tokens.
"""

import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request

from . import pointers, telemetry, tools as tools_mod

DEFAULT_EMBED = "https://ramzesx--v3-code-embeddings-serve.modal.run"
DEFAULT_RERANK = "https://ramzesx--v3-code-reranker-serve.modal.run"
BATCH = 48
CANDIDATES = 30
_WORD_RX = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def split_camel(s):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s).replace("_", " ").replace("-", " ")


def socket_text(ent, nav):
    """The short text an entity is searchable by: what a newcomer would type, not the code."""
    repo, rel = pointers.split_path(ent.get("file_path", ""))
    stem = ent.get("name", "").rsplit(".", 1)[0]
    parts = [p for p in rel.split("/")[:-1] if p not in ("src", "main", "java", "com", "app", "test")]
    words = " ".join(split_camel(p) for p in parts[-4:])
    bits = [f"{split_camel(stem)} ({ent.get('entity_type')} in {repo} {words}).",
            f"subsystem {nav.get('name')}" if nav else ""]
    if nav and nav.get("responsibilities"):
        resp = nav["responsibilities"]
        if isinstance(resp, str):
            try:
                resp = json.loads(resp)
            except ValueError:
                resp = [resp]
        bits.append("; ".join(str(r) for r in list(resp)[:4]))
    if str(ent.get("entry_point", "")).lower() == "true":
        bits.append("entry point")
    return " ".join(b for b in bits if b)[:600]


class ModalClient:
    def __init__(self, embed_url=None, rerank_url=None, http=None, timeout=200, poll_max_s=360):
        self.embed_url = embed_url or os.environ.get("MODAL_EMBED_URL") or DEFAULT_EMBED
        self.rerank_url = rerank_url or os.environ.get("MODAL_RERANK_URL") or DEFAULT_RERANK
        self.http = http  # tests: (method, url, body) -> (status, headers, text)
        self.timeout = timeout
        self.poll_max_s = poll_max_s

    def _request(self, method, url, body=None):
        if self.http is not None:
            return self.http(method, url, body)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # NOSONAR - configured https endpoint; see sonar-project.properties
                return r.status, dict(r.headers), r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read().decode("utf-8", "replace")

    def call(self, url, body):
        """POST, following Modal's 303 long-request redirect until the result is ready."""
        status, headers, text = self._request("POST", url, body)
        t0 = time.time()
        while status in (303, 202) and time.time() - t0 < self.poll_max_s:
            loc = headers.get("Location") or headers.get("location")
            if not loc:
                break
            time.sleep(2)
            status, headers, text = self._request("GET", loc)
        if status != 200:
            raise RuntimeError(f"modal {status}: {text[:200]}")
        return json.loads(text)

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), BATCH):
            res = self.call(self.embed_url, {"texts": texts[i:i + BATCH]})
            vecs = res.get("embeddings") if isinstance(res, dict) else res
            out.extend(vecs)
        return out

    def rerank(self, query, documents):
        res = self.call(self.rerank_url, {"query": query, "documents": documents})
        return res.get("scores") if isinstance(res, dict) else res


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class SearchIndex:
    def __init__(self, app, client=None, cache_dir=None):
        self.app = app
        self.client = client or ModalClient()
        self.cache_dir = cache_dir or os.environ.get("CODEMAP_TELEMETRY_DIR") or telemetry.DEFAULT_DIR
        self.names = []
        self.texts = []
        self.vecs = []
        self.state = "empty"  # empty | building | ready | failed
        self.error = None
        self._lock = threading.Lock()
        self._thread = None

    @property
    def cache_path(self):
        return os.path.join(self.cache_dir, f"search_index-{self.app.pack_version.replace('@', '-')}.jsonl")

    def sockets(self):
        rows = []
        for ent in self.app.engine.ents:
            sub = ent.get("curated") or ent.get("subsystem")
            nav = self.app.clue_full(sub) if hasattr(self.app, "clue_full") else None
            rows.append((ent["name"], socket_text(ent, nav)))
        return rows

    def load(self):
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.names.append(row["name"])
                    self.texts.append(row["text"])
                    self.vecs.append(row["vec"])
            if self.names:
                self.state = "ready"
                return True
        except (OSError, ValueError):
            pass
        return False

    def build(self):
        with self._lock:
            if self.state in ("building", "ready"):
                return self.state
            self.state = "building"
        try:
            rows = self.sockets()
            vecs = self.client.embed([t for _, t in rows])
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                for (name, text), vec in zip(rows, vecs):
                    f.write(json.dumps({"name": name, "text": text, "vec": [round(x, 6) for x in _norm(vec)]}) + "\n")
            self.names = [n for n, _ in rows]
            self.texts = [t for _, t in rows]
            self.vecs = [_norm(v) for v in vecs]
            self.state = "ready"
        except Exception as ex:  # a failed build is a state, not a crash
            self.state, self.error = "failed", f"{type(ex).__name__}: {str(ex)[:200]}"
        return self.state

    def build_async(self):
        self._thread = threading.Thread(target=self.build, name="codemap-search-index", daemon=True)
        self._thread.start()

    def wait(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)
        return self.state

    def query(self, text, k=5):
        qv = _norm(self.client.embed([text])[0])
        scored = sorted(((sum(a * b for a, b in zip(qv, v)), i) for i, v in enumerate(self.vecs)), reverse=True)[:CANDIDATES]
        cand = [i for _, i in scored]
        scores = self.client.rerank(text, [self.texts[i] for i in cand])
        ranked = sorted(zip(scores, cand), key=lambda x: -x[0])[:k]
        return [(self.names[i], float(s), float(dict((j, c) for c, j in scored)[i])) for s, i in ranked]


def search(app, ident, args):
    text = (args.get("text") or "").strip()
    k = args.get("k")
    k = 5 if k is None else k
    if not text:
        return json.dumps({"error": "text required"}), True
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 20:
        return json.dumps({"error": "k must be an integer 1-20"}), True
    idx = getattr(app, "search_index", None)
    if idx is None or idx.state != "ready":
        state = idx.state if idx else "disabled"
        if idx is not None and idx.state == "empty":
            idx.build_async()
            state = "building"
        return json.dumps({"error": f"search index {state}; use codemap_step find(term) meanwhile",
                           "detail": (idx.error if idx else None)}), True
    st = app.budget_state(ident.user)
    if st["exhausted"]:
        app.emit(tools_mod.base_event(app, ident, "budget_refusal", "search", terminal="budget_exhausted", q=text,
                                      credits=0.0, tool="codemap_search", spent=st["spent"], budget=st["budget"]))
        return json.dumps({"error": "budget_exhausted", "terminal": "budget_exhausted", "retry_after_s": st["resets_in_s"]}), True
    try:
        hits = idx.query(text, k)
    except Exception as ex:
        app.emit(tools_mod.base_event(app, ident, "search", "search", terminal="error", q=text, credits=0.0,
                                      tool="codemap_search", error=f"{type(ex).__name__}: {str(ex)[:200]}"))
        return json.dumps({"error": f"search failed: {type(ex).__name__}"}), True
    tokens = {"prompt": max(1, len(text) // 4), "completion": 0, "cached": 0, "cache_creation": 0}
    credits = app.credits_for("search", None, tokens, flat="search_call")
    ptrs = []
    for name, rerank_score, cos in hits:
        ent = app.entity(name)
        if ent:
            p = pointers.to_pointer(ent, clue=app.clue(ent.get("curated") or ent.get("subsystem")))
            p["score"] = round(rerank_score, 4)
            p["cosine"] = round(cos, 4)
            ptrs.append(p)
    app.emit(tools_mod.base_event(app, ident, "search", "search", terminal="answer", q=text, pointers=ptrs,
                                  credits=credits, tokens=tokens, tool="codemap_search", steps=1))
    return json.dumps({"pointers": ptrs, "request_id": ident.request_id, "credits": credits,
                       "note": "reranked by Qwen3-Reranker-8B; open the top pointer, then codemap_feedback"}, ensure_ascii=False), False
