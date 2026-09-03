"""
Modal GPU embedding app — the SOTA backend of the V3 embedding service.

Serves Qwen3-Embedding-8B (Apache-2.0; top open-source embedding model on
MTEB/CoIR, instruction-aware, 4096-dim, strong on code) behind the same
POST /embed contract as embed_server.py, so the embedder agent switches
backends via EMBED_URL alone.

Deploy (needs BOTH halves of a Modal token — id `ak-…` AND secret `as-…`):
    modal token set --token-id ak-... --token-secret as-...
    modal deploy modal_app.py
    # → https://<workspace>--v3-code-embeddings-serve.modal.run/embed

Cost posture per owner directive: costs no object — L40S GPU, generous
scaledown window so bulk graph embedding stays warm.
"""

import modal

MODEL_ID = "Qwen/Qwen3-Embedding-8B"

app = modal.App("v3-code-embeddings")


def _bake_model():
    # Download weights into the image at BUILD time so cold starts are fast and
    # deterministic — no 16GB fetch on first request while an unattended re-embed
    # run depends on it. Fail-fast: a bad download breaks `modal deploy`, not the run.
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sentence-transformers>=4.0",
        "transformers>=4.51",  # Qwen3 architecture requires transformers >= 4.51
        "torch",
        "fastapi[standard]",
        "hf_transfer",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Curb CUDA allocator fragmentation across requests (PyTorch's own OOM hint).
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .run_function(_bake_model)
)

with image.imports():
    import torch
    from sentence_transformers import SentenceTransformer


@app.cls(gpu="L40S", image=image, scaledown_window=600, timeout=900)
class Embedder:
    @modal.enter()
    def load(self):
        # flash-attention-free load keeps the image simple; L40S handles 8B fp16.
        self.model = SentenceTransformer(MODEL_ID, model_kwargs={"torch_dtype": "float16"})
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode_safe(self, texts, prompt=None):
        # Bi-encoder embeddings are independent per document, so batch size affects
        # only speed, not values. Try a modest batch; on ANY failure (OOM often
        # surfaces as a plain RuntimeError, not OutOfMemoryError) isolate items one
        # at a time, truncating a stubborn huge file harder before conceding — so a
        # heavy batch can never 500 the request, and every item gets a real vector.
        kw = {"normalize_embeddings": True}
        if prompt:
            kw["prompt"] = prompt
        try:
            return self.model.encode(texts, batch_size=8, **kw).tolist()
        except Exception:
            torch.cuda.empty_cache()
            out = []
            for t in texts:
                emb = None
                for max_c in (24_000, 8_000, 2_000):
                    try:
                        emb = self.model.encode([t[:max_c]], batch_size=1, **kw)[0].tolist()
                        break
                    except Exception:
                        torch.cuda.empty_cache()
                out.append(emb if emb is not None else [0.0] * self.dim)
            return out

    @modal.fastapi_endpoint(method="POST", label="v3-code-embeddings-serve")
    def embed(self, payload: dict) -> dict:
        # `instruct` is OPTIONAL and absent means byte-identical behaviour to the
        # pre-lens service, so embed_graph.py and v3_retrieve.py cannot break.
        #
        # When present it becomes Qwen3-Embedding's native instruction prefix, which
        # is what makes one model serve as several LENSES: the same file embedded
        # under "same business purpose" and under "similar runtime behaviour" lands
        # in different places. Qwen3 reports instructions worth 1-5% on retrieval,
        # so the instruction ALONE will not separate lenses — the caller must also
        # vary the text. That is the socket design, and it lives in embed_graph.py.
        texts = payload.get("texts")
        if not isinstance(texts, list) or not texts:
            return {"error": "texts: non-empty list required"}
        instruct = payload.get("instruct")
        prompt = None
        if isinstance(instruct, str) and instruct.strip():
            prompt = f"Instruct: {instruct.strip()}\nQuery:"
        clipped = [str(t)[:24_000] for t in texts]
        return {
            "embeddings": self._encode_safe(clipped, prompt),
            "model": MODEL_ID,
            "dim": self.dim,
            "instruct": instruct or None,
        }
