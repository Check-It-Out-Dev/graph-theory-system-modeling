"""
Modal GPU reranker — Qwen3-Reranker-8B cross-encoder, the precision half of the
V3 two-stage retrieval stack.

The embedder (Qwen3-Embedding-8B, embed_modal_app / modal_app.py) does fast kNN
*recall* over the graph's 4096-dim vectors. This reranker does precise *ranking*:
it scores each (query, document) pair so Erdos's "what is wrong" hunt reasons over
the genuinely most-relevant code, not merely the nearest neighbours. Two-stage
retrieve-then-rerank is the SOTA RAG pattern.

Contract:  POST {"query": str, "documents": [str, ...], "instruction": str?}
        -> {"scores": [float, ...], "model": str}       # score in [0,1] = P(relevant)

Scoring follows the Qwen3-Reranker model card exactly: format the pair into the
yes/no judging prompt, read the final-token logits, softmax over the "yes"/"no"
tokens, take P(yes).

Deploy:  PYTHONUTF8=1 modal deploy reranker_modal_app.py
"""

import modal

MODEL_ID = "Qwen/Qwen3-Reranker-8B"
MAX_LEN = 8192
CHUNK = 8  # sub-batch so a large rerank set can't OOM the 8B model on one forward

app = modal.App("v3-code-reranker")


def _bake_model():
    # Bake weights at build time — deterministic, fast cold starts for unattended use.
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers>=4.51", "torch", "fastapi[standard]", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(_bake_model)
)

with image.imports():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


@app.cls(gpu="L40S", image=image, scaledown_window=600, timeout=900)
class Reranker:
    @modal.enter()
    def load(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        self.model = (
            AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
            .eval()
            .cuda()
        )
        self.tok_yes = self.tok.convert_tokens_to_ids("yes")
        self.tok_no = self.tok.convert_tokens_to_ids("no")
        self.prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements "
            'based on the Query and the Instruct provided. Note that the answer can '
            'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_ids = self.tok.encode(self.prefix, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(self.suffix, add_special_tokens=False)

    def _format(self, instruction, query, doc):
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _score_chunk(self, pairs):
        budget = MAX_LEN - len(self.prefix_ids) - len(self.suffix_ids)
        enc = self.tok(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=budget,
        )
        for i, ids in enumerate(enc["input_ids"]):
            enc["input_ids"][i] = self.prefix_ids + ids + self.suffix_ids
        enc = self.tok.pad(enc, padding=True, return_tensors="pt", max_length=MAX_LEN)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits[:, -1, :]
            stacked = torch.stack([logits[:, self.tok_no], logits[:, self.tok_yes]], dim=1)
            return torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp().tolist()

    @modal.fastapi_endpoint(method="POST", label="v3-code-reranker-serve")
    def rerank(self, payload: dict) -> dict:
        query = payload.get("query")
        docs = payload.get("documents")
        if not isinstance(query, str) or not isinstance(docs, list) or not docs:
            return {"error": "need query:str and documents:non-empty list"}
        instruction = payload.get(
            "instruction",
            "Given a code-investigation query, retrieve the most relevant source files",
        )
        pairs = [self._format(instruction, query, str(d)[:24_000]) for d in docs]
        scores = []
        for start in range(0, len(pairs), CHUNK):
            scores.extend(self._score_chunk(pairs[start : start + CHUNK]))
        return {"scores": scores, "model": MODEL_ID}
