"""
V3 embedder agent — fills the embeddings Hypatia's indexing pass doesn't
(she classifies + wires arrows; vectors need an embedding backend).

For every node in the target namespace that has a source `path` but no
`embedding`, this agent reads the file, embeds it through the /embed
service (local fastembed server or the Modal Qwen3-8B app — same contract,
pick via EMBED_URL), and writes back:
    n.embedding        list<float>
    n.embedding_model  provenance (mixed-model states stay detectable)
Then it ensures a cosine vector index over the namespace's label.

Idempotent: re-running only touches nodes still missing `embedding`.
Safety: writes ONLY to nodes in NAMESPACE; never touches other namespaces.

Usage:
    python embed_graph.py                     # full run over CheckItOutV3
    python embed_graph.py --limit 5 --dry     # smoke: embed 5, report, no index
"""

import argparse
import os
import sys
import time

import requests
from neo4j import GraphDatabase

EMBED_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:8899/embed")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7611")
NEO4J_AUTH = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASS", "password"))
NAMESPACE = os.environ.get("EMBED_NAMESPACE", "CheckItOutV3")
BATCH = 16
MAX_CHARS = 24_000

PENDING_QUERY = """
MATCH (n {namespace: $ns})
WHERE coalesce(n.file_path, n.path) IS NOT NULL
  AND n.embedding IS NULL
  AND coalesce(n.embedding_status, 'PENDING') = 'PENDING'
RETURN elementId(n) AS eid, coalesce(n.file_path, n.path) AS path, coalesce(n.name,'') AS name
LIMIT $limit
"""

WRITE_QUERY = """
UNWIND $rows AS row
MATCH (n) WHERE elementId(n) = row.eid
SET n.embedding = row.vector, n.embedding_model = row.model, n.embedding_status = 'DONE'
"""


def read_snippet(path: str, name: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read(MAX_CHARS)
        # Prefix the identity line — retrieval queries mention names/paths.
        return f"// {name} — {path}\n{content}"
    except OSError:
        return f"// {name} — {path}\n(unreadable source file)"


def embed(texts: list[str], retries: int = 6) -> tuple[list[list[float]], str]:
    # Resilient POST: Modal cold-start + transient network/5xx must not kill an
    # unattended run. Retry with capped exponential backoff; raise only if all fail.
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(EMBED_URL, json={"texts": texts}, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            if "embeddings" not in data:
                raise RuntimeError(f"embed service error: {data}")
            return data["embeddings"], data.get("model", "unknown")
        except Exception as exc:  # noqa: BLE001 — network/HTTP/JSON/service, all retryable
            last_err = exc
            wait = min(30, 2 ** attempt)
            print(
                f"[embedder] batch attempt {attempt + 1}/{retries} failed: {exc}; "
                f"retry in {wait}s",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"batch failed after {retries} retries: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100_000)
    ap.add_argument("--dry", action="store_true", help="skip vector-index creation")
    args = ap.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    done = 0
    errors = 0
    dim = None
    model_used = None
    t0 = time.time()
    with driver.session() as session:
        # Self-heal: drop any prior-dimension index so 4096-dim writes never hit a
        # dimension mismatch. Recreated at the current dimension after embedding.
        session.run("DROP INDEX v3_code_embedding IF EXISTS")
        while done + errors < args.limit:
            page = min(BATCH, args.limit - done - errors)
            rows = session.run(PENDING_QUERY, ns=NAMESPACE, limit=page).data()
            if not rows:
                break
            texts = [read_snippet(r["path"], r["name"]) for r in rows]
            try:
                vectors, model_used = embed(texts)
            except RuntimeError as exc:
                # embed() already exhausted its retries — mark ERROR so these nodes
                # drop out of PENDING and the run keeps moving (re-run picks them up).
                print(
                    f"[embedder] batch permanently failed, marking {len(rows)} ERROR: {exc}",
                    flush=True,
                )
                session.run(
                    "UNWIND $eids AS eid MATCH (n) WHERE elementId(n)=eid "
                    "SET n.embedding_status='ERROR'",
                    eids=[r["eid"] for r in rows],
                )
                errors += len(rows)
                continue
            dim = len(vectors[0])
            session.run(
                WRITE_QUERY,
                rows=[
                    {"eid": r["eid"], "vector": v, "model": model_used}
                    for r, v in zip(rows, vectors)
                ],
            )
            done += len(rows)
            print(
                f"[embedder] {done} embedded, {errors} errored ({time.time() - t0:.0f}s)",
                flush=True,
            )

        remaining = session.run(
            "MATCH (n {namespace:$ns}) WHERE coalesce(n.file_path, n.path) IS NOT NULL "
            "AND n.embedding IS NULL AND coalesce(n.embedding_status,'PENDING') = 'PENDING' "
            "RETURN count(n) AS c",
            ns=NAMESPACE,
        ).single()["c"]

        if not args.dry and dim:
            # Hypatia's V3 file nodes carry the EntityDetail label.
            session.run(
                f"CREATE VECTOR INDEX v3_code_embedding IF NOT EXISTS "
                f"FOR (n:EntityDetail) ON (n.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, "
                f"`vector.similarity_function`: 'cosine'}}}}"
            )

    driver.close()
    print(
        f"[embedder] DONE: {done} embedded, {errors} errored, {remaining} still pending, "
        f"model={model_used}, dim={dim}, {time.time() - t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
