"""
V3 two-stage semantic retrieval over the CheckItOutV3 code graph.

Stage 1 (recall):    embed the query with Qwen3-Embedding-8B, kNN over the graph's
                     4096-dim vectors (Neo4j vector index `v3_code_embedding`).
Stage 2 (precision): rerank the candidates with Qwen3-Reranker-8B (cross-encoder).

Gives Erdos (and the operator) a natural-language search over the actual code —
the capability the base V3 prompts lack, aimed at the "what is wrong" hunt:

    python v3_retrieve.py "transaction boundary crossed with a lazy Hibernate proxy" --top 10
    python v3_retrieve.py "where is rate limiting enforced" --top 8 --recall 50

Env: EMBED_URL, RERANK_URL, NEO4J_URI/USER/PASS, EMBED_NAMESPACE.
"""

import argparse
import os
import sys

import requests
from neo4j import GraphDatabase

EMBED_URL = os.environ.get("EMBED_URL", "https://ramzesx--v3-code-embeddings-serve.modal.run")
RERANK_URL = os.environ.get("RERANK_URL", "https://ramzesx--v3-code-reranker-serve.modal.run")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7611")
NEO4J_AUTH = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASS", "password"))
NAMESPACE = os.environ.get("EMBED_NAMESPACE", "CheckItOutV3")
MAX_CHARS = 24_000

# Qwen3-Embedding is asymmetric: the query gets an instruction, documents stay plain
# (the graph vectors were embedded plain). This matches the model's training.
QUERY_INSTRUCTION = "Given a code-investigation query, retrieve the most relevant source files"


def embed_query(q: str) -> list[float]:
    text = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {q}"
    r = requests.post(EMBED_URL, json={"texts": [text]}, timeout=300)
    r.raise_for_status()
    return r.json()["embeddings"][0]


def knn(session, vec: list[float], k: int) -> list[dict]:
    return session.run(
        """
        CALL db.index.vector.queryNodes('v3_code_embedding', $k, $vec)
        YIELD node, score
        WHERE node.namespace = $ns
        RETURN coalesce(node.name,'') AS name,
               coalesce(node.file_path, node.path,'') AS path,
               coalesce(node.entity_type, '') AS etype,
               score AS knn_score
        """,
        k=k,
        vec=vec,
        ns=NAMESPACE,
    ).data()


def read_snippet(path: str, name: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return f"// {name} - {path}\n{fh.read(MAX_CHARS)}"
    except OSError:
        return f"// {name} - {path}\n(unreadable source)"


def rerank(query: str, docs: list[str]) -> list[float]:
    r = requests.post(RERANK_URL, json={"query": query, "documents": docs}, timeout=600)
    r.raise_for_status()
    return r.json()["scores"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top", type=int, default=10, help="final results after rerank")
    ap.add_argument("--recall", type=int, default=40, help="kNN candidates before rerank")
    ap.add_argument("--no-rerank", action="store_true", help="kNN only (skip stage 2)")
    args = ap.parse_args()

    vec = embed_query(args.query)
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session() as session:
        cands = knn(session, vec, args.recall)
    driver.close()
    if not cands:
        print("No candidates — is the vector index `v3_code_embedding` built?")
        return 1

    if args.no_rerank:
        ranked = sorted(cands, key=lambda c: c["knn_score"], reverse=True)
        key = "knn_score"
    else:
        docs = [read_snippet(c["path"], c["name"]) for c in cands]
        for c, s in zip(cands, rerank(args.query, docs)):
            c["rerank"] = s
        ranked = sorted(cands, key=lambda c: c["rerank"], reverse=True)
        key = "rerank"

    print(f"# query: {args.query}")
    print(f"# recall={args.recall} -> {'rerank' if not args.no_rerank else 'knn'} -> top {args.top}\n")
    for i, c in enumerate(ranked[: args.top], 1):
        print(f"{i:2d}. {c[key]:.3f}  {c['etype']:9s} {c['name']:38s} {c['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
