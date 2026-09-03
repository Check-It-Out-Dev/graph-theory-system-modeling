# V3 Embedding + Retrieval Service

SOTA semantic layer for the CheckItOutV3 code graph (Neo4j `bolt://127.0.0.1:7611`,
namespace `CheckItOutV3`, node label `EntityDetail`). Two Qwen3-8B models on Modal
GPU (L40S) power embedding and two-stage retrieval.

## Deployed Modal services (workspace `ramzesx`)

| Service | Model | Endpoint | Contract |
|---|---|---|---|
| Embedder | `Qwen/Qwen3-Embedding-8B` (4096-dim) | `https://ramzesx--v3-code-embeddings-serve.modal.run` | `POST {"texts":[...]}` → `{"embeddings","model","dim"}` |
| Reranker | `Qwen/Qwen3-Reranker-8B` (cross-encoder) | `https://ramzesx--v3-code-reranker-serve.modal.run` | `POST {"query","documents":[...],"instruction"?}` → `{"scores","model"}` |

Both bake weights into the image (fast, deterministic cold starts ~40s), run
`expandable_segments` to avoid CUDA fragmentation, and degrade gracefully on OOM
(embedder falls back to item-by-item with progressive truncation — never 500s).

Redeploy: `PYTHONUTF8=1 modal deploy modal_app.py` (or `reranker_modal_app.py`).
The `PYTHONUTF8=1` prefix is required on Windows or Modal's rich output crashes on cp1252.

## Files

- `modal_app.py` — embedder service (Qwen3-Embedding-8B).
- `reranker_modal_app.py` — reranker service (Qwen3-Reranker-8B).
- `embed_graph.py` — Neo4j-driven embedder: reads each node's source file, embeds via
  `EMBED_URL`, writes `embedding`/`embedding_model`/`embedding_status='DONE'`, rebuilds the
  `v3_code_embedding` vector index (4096-dim cosine). Retry+backoff, ERROR-marking,
  self-healing index. Idempotent — re-run to top up. `--limit N`, `--dry`.
- `v3_retrieve.py` — two-stage semantic search (the instrument for the "what is wrong" hunt).
- `embed_server.py` — local CPU fallback embedder (jina-code-v2, 768-dim). Not used when Modal is up.

## Two-stage retrieval (Erdos's hunt tool)

```
python v3_retrieve.py "<natural-language code concern>" --top 10 --recall 40
```

Stage 1 embeds the query (Qwen3-Embedding-8B) and pulls the `--recall` nearest nodes
from the `v3_code_embedding` index; stage 2 reranks them (Qwen3-Reranker-8B) and prints
the `--top` by relevance. `--no-rerank` for kNN only.

Proven: query *"transaction boundary crossed with a lazy Hibernate proxy…"* ranks the
LazyInit fetch-join integration test #1 (0.989), then the lazy-proxy service and the
repository holding the `LEFT JOIN FETCH` fix — exactly the relevant code.

## Env (all have sane defaults)

`EMBED_URL`, `RERANK_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`, `EMBED_NAMESPACE`.

## Modal auth

`modal token set --token-id ak-… --token-secret as-…` (workspace `ramzesx`). Tokens live
in `~/.modal.toml` — never committed.
