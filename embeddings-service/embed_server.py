"""
Local code-embedding HTTP server — the CPU fallback backend of the V3
embedding service (same /embed contract as the Modal GPU app in
modal_app.py, so the embedder agent is backend-agnostic).

Model: jinaai/jina-embeddings-v2-base-code via fastembed (ONNX, CPU-fast) —
the strongest open-source CODE embedding model that runs well without a GPU
(768-dim, 8192-token context, trained on code+docstrings). The Modal app
serves Qwen3-Embedding-8B (4096-dim) when GPU SOTA is wanted; the graph
stores whichever model produced the vector in `embedding_model` so mixed
states are detectable.

Run:  python embed_server.py  [PORT=8899]
Test: curl -s -X POST http://127.0.0.1:8899/embed -H "Content-Type: application/json" \
        -d '{"texts":["def add(a,b): return a+b"]}'
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastembed import TextEmbedding

MODEL_NAME = os.environ.get("EMBED_MODEL", "jinaai/jina-embeddings-v2-base-code")
PORT = int(os.environ.get("EMBED_PORT", "8899"))
MAX_CHARS = 24_000  # ~8k tokens of code; fastembed truncates internally too

print(f"[embed-server] loading {MODEL_NAME} (first run downloads ONNX weights)...")
t0 = time.time()
_model = TextEmbedding(model_name=MODEL_NAME)
_dim = len(next(iter(_model.embed(["warmup"]))))
print(f"[embed-server] ready in {time.time() - t0:.1f}s — dim={_dim} port={PORT}")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 — stdlib naming
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": MODEL_NAME, "dim": _dim})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/embed":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length))
            texts = data.get("texts")
            if not isinstance(texts, list) or not texts:
                self._send(400, {"error": "texts: non-empty list required"})
                return
            clipped = [str(t)[:MAX_CHARS] for t in texts]
            vectors = [v.tolist() for v in _model.embed(clipped, batch_size=8)]
            self._send(200, {"embeddings": vectors, "model": MODEL_NAME, "dim": _dim})
        except Exception as exc:  # surface the reason to the caller, keep serving
            self._send(500, {"error": str(exc)})

    def log_message(self, fmt, *args):  # quiet per-request noise
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
