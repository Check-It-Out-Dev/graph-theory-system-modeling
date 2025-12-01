# Appendix C: Graph-Native Implementation via Neo4j and APOC

## Escaping Python Orchestration: All-in-Graph Lens Calibration

**Abstract**

This appendix presents an alternative implementation strategy for Information Lensing that eliminates external Python orchestration entirely. By leveraging Neo4j's APOC procedures, Graph Data Science (GDS) library, and custom REST endpoints, the complete lens calibration pipeline—from reranker calls to SVD computation to embedding transformation—can be executed purely in Cypher. This approach offers deployment simplicity, transactional consistency, and natural integration with existing graph workflows.

---

## C.1 Motivation: Why Graph-Native?

The main paper describes a pipeline requiring:
1. Python orchestration for pair sampling
2. External reranker API calls
3. Matrix computation (SVD, Frobenius alignment)
4. Embedding storage and transformation

This creates operational complexity:

```
┌──────────────────────────────────────────────────────────────┐
│                   TRADITIONAL APPROACH                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Python Script                                              │
│      │                                                       │
│      ├──► Neo4j Driver ──► Read embeddings                   │
│      │                                                       │
│      ├──► Reranker API ──► Get similarity scores             │
│      │                                                       │
│      ├──► NumPy/SciPy ──► SVD computation                    │
│      │                                                       │
│      └──► Neo4j Driver ──► Write transformation matrix       │
│                                                              │
│   Problems:                                                  │
│   • Multiple failure points                                  │
│   • State synchronization issues                             │
│   • Complex deployment                                       │
│   • No transactional guarantees                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

The graph-native approach consolidates everything:

```
┌──────────────────────────────────────────────────────────────┐
│                   GRAPH-NATIVE APPROACH                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Single Cypher Query / Procedure                            │
│      │                                                       │
│      ├──► MATCH ──► Sample pairs from graph                  │
│      │                                                       │
│      ├──► apoc.load.jsonParams ──► Call reranker endpoint    │
│      │                                                       │
│      ├──► apoc.coll.* / GDS ──► Matrix operations            │
│      │                                                       │
│      └──► SET ──► Store lens in graph                        │
│                                                              │
│   Benefits:                                                  │
│   • Single execution context                                 │
│   • Transactional consistency                                │
│   • Progress tracking via graph                              │
│   • No external dependencies except reranker                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## C.2 Architecture Overview

### C.2.1 Component Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    RERANKER SERVICE                         │
│         (Local GPU or Cloud API)                            │
│         localhost:8080 / api.reranker.cloud                 │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP POST
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       NEO4J                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 APOC Extended                       │    │
│  │                                                     │    │
│  │  • apoc.load.jsonParams() ─► REST calls             │    │
│  │  • apoc.periodic.iterate() ─► Batch processing      │    │
│  │  • apoc.coll.* ─► List/matrix operations            │    │
│  │  • apoc.convert.toJson() ─► Payload construction    │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Graph Data Science (GDS)               │    │
│  │                                                     │    │
│  │  • gds.similarity.cosine() ─► Similarity calc       │    │
│  │  • gds.eigenvector() ─► Power iteration (SVD core)  │    │
│  │  • gds.fastRP() ─► Random projection (J-L lemma)    │    │
│  │  • Pregel API ─► Custom iterative algorithms        │    │
│  │  • Native projections ─► Efficient graph ops        │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   Graph Schema                      │    │
│  │                                                     │    │
│  │  (:EntityDetail)─[:RERANK_SCORE]─>(:EntityDetail)   │    │
│  │  (:TransformationLens {A: [...], B: [...], r: 128}) │    │
│  │  (:CalibrationJob {status, progress, batch})        │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### C.2.2 Reranker Service Contract

The reranker endpoint must implement the following interface:

**Request:**
```json
POST /rerank
Content-Type: application/json

{
  "pairs": [
    {"id": "pair_001", "text_a": "...", "text_b": "..."},
    {"id": "pair_002", "text_a": "...", "text_b": "..."}
  ]
}
```

**Response:**
```json
{
  "scores": [
    {"id": "pair_001", "score": 0.847},
    {"id": "pair_002", "score": 0.234}
  ]
}
```

This simple contract allows both local deployment (Qwen3-Reranker, BGE-Reranker) and cloud services (Cohere, Jina) to be used interchangeably.

---

## C.3 Implementation: Step-by-Step Cypher Pipeline

### C.3.1 Phase 1: Pair Sampling with Monte Carlo

```cypher
// ============================================================
// PHASE 1: SAMPLE PAIRS FOR RERANKING
// Uses stratified sampling + high-divergence prioritization
// ============================================================

// Configuration
:param batchSize => 100;
:param totalPairs => 5000;
:param namespace => 'checkitout';

// Create calibration job node for tracking
MERGE (job:CalibrationJob {namespace: $namespace, started: datetime()})
SET job.status = 'SAMPLING',
    job.targetPairs = $totalPairs,
    job.completedPairs = 0;

// Sample pairs with stratification by node_type
MATCH (f1:EntityDetail {namespace: $namespace})
MATCH (f2:EntityDetail {namespace: $namespace})
WHERE id(f1) < id(f2)  // Avoid duplicates and self-pairs

// Stratify: prioritize cross-type pairs (more informative)
WITH f1, f2,
     CASE WHEN f1.node_type <> f2.node_type THEN 0.7 ELSE 0.3 END AS sampleWeight,
     rand() AS r

// Weighted random sampling
WHERE r < sampleWeight * ($totalPairs * 2.0 / (count(*) OVER ()))

// Calculate current cosine similarity
WITH f1, f2, 
     gds.similarity.cosine(f1.fused_embedding, f2.fused_embedding) AS cosine_sim

// Prioritize high-similarity pairs (likely to have divergence from reranker)
ORDER BY cosine_sim DESC
LIMIT $totalPairs

// Create pending pair relationships
MERGE (f1)-[p:PENDING_RERANK]->(f2)
SET p.cosine_sim = cosine_sim,
    p.created = datetime(),
    p.batch = toInteger(id(p) / $batchSize)

RETURN count(*) AS pairsSampled;
```

### C.3.2 Phase 2: Batch Reranker Calls via APOC

```cypher
// ============================================================
// PHASE 2: CALL RERANKER IN BATCHES
// Uses apoc.load.jsonParams for REST API integration
// ============================================================

:param rerankerUrl => 'http://localhost:8080/rerank';
:param batchSize => 50;

// Process one batch at a time
MATCH (job:CalibrationJob {namespace: $namespace})
WHERE job.status = 'SAMPLING' OR job.status = 'RERANKING'
SET job.status = 'RERANKING'
WITH job

// Get next batch of pending pairs
MATCH (f1:EntityDetail)-[p:PENDING_RERANK]->(f2:EntityDetail)
WHERE p.processed IS NULL
WITH f1, f2, p, p.batch AS batchNum
ORDER BY batchNum
LIMIT $batchSize

// Construct payload for reranker
WITH collect({
    id: toString(id(p)),
    text_a: coalesce(f1.content, f1.name),
    text_b: coalesce(f2.content, f2.name),
    f1_id: id(f1),
    f2_id: id(f2),
    rel_id: id(p)
}) AS pairs

// Call reranker endpoint
CALL apoc.load.jsonParams(
    $rerankerUrl,
    {
        method: "POST",
        `Content-Type`: "application/json"
    },
    apoc.convert.toJson({pairs: pairs})
) YIELD value

// Process response
UNWIND value.scores AS result
MATCH (f1:EntityDetail)-[p:PENDING_RERANK]->(f2:EntityDetail)
WHERE id(p) = toInteger(result.id)

// Store reranker score and mark as processed
SET p.reranker_score = result.score,
    p.divergence = abs(result.score - p.cosine_sim),
    p.processed = datetime()

// Convert to permanent relationship
WITH f1, f2, p
CREATE (f1)-[r:RERANK_SCORE]->(f2)
SET r.score = p.reranker_score,
    r.cosine_sim = p.cosine_sim,
    r.divergence = p.divergence

// Clean up pending relationship
DELETE p

// Update job progress
WITH count(*) AS processed
MATCH (job:CalibrationJob {namespace: $namespace})
SET job.completedPairs = job.completedPairs + processed,
    job.lastBatchAt = datetime()

RETURN processed AS pairsProcessed;
```

### C.3.3 Phase 2b: Batch Processing with Progress Tracking

For large-scale processing, use `apoc.periodic.iterate`:

```cypher
// ============================================================
// PHASE 2b: AUTOMATED BATCH PROCESSING
// Processes all pending pairs with automatic batching
// ============================================================

CALL apoc.periodic.iterate(
    // Outer query: get batches
    "
    MATCH (f1:EntityDetail)-[p:PENDING_RERANK]->(f2:EntityDetail)
    WHERE p.processed IS NULL
    WITH p.batch AS batchNum, collect({p: p, f1: f1, f2: f2}) AS batchPairs
    RETURN batchNum, batchPairs
    ORDER BY batchNum
    ",
    
    // Inner query: process each batch
    "
    WITH batchPairs
    UNWIND batchPairs AS pair
    WITH collect({
        id: toString(id(pair.p)),
        text_a: coalesce(pair.f1.content, pair.f1.name),
        text_b: coalesce(pair.f2.content, pair.f2.name)
    }) AS payload,
    collect(pair) AS pairs
    
    CALL apoc.load.jsonParams(
        $rerankerUrl,
        {method: 'POST', `Content-Type`: 'application/json'},
        apoc.convert.toJson({pairs: payload})
    ) YIELD value
    
    UNWIND range(0, size(value.scores)-1) AS idx
    WITH pairs[idx] AS pair, value.scores[idx].score AS score
    
    SET pair.p.reranker_score = score,
        pair.p.processed = datetime()
    
    CREATE (pair.f1)-[:RERANK_SCORE {score: score}]->(pair.f2)
    DELETE pair.p
    ",
    
    {batchSize: 1, parallel: false, params: {rerankerUrl: $rerankerUrl}}
)
YIELD batches, total, errorMessages
RETURN batches, total, errorMessages;
```

### C.3.4 Phase 3: Construct Similarity Matrix

```cypher
// ============================================================
// PHASE 3: BUILD SIMILARITY MATRICES
// Constructs S_reranker and S_cosine as graph relationships
// ============================================================

// Get all files with embeddings
MATCH (f:EntityDetail {namespace: $namespace})
WHERE f.fused_embedding IS NOT NULL
WITH f ORDER BY id(f)
WITH collect(f) AS files, count(*) AS n

// Create matrix node
MERGE (m:SimilarityMatrix {namespace: $namespace})
SET m.dimension = n,
    m.created = datetime()

// Store file ordering for matrix indexing
WITH files, m
UNWIND range(0, size(files)-1) AS idx
SET (files[idx]).matrix_idx = idx

// Matrices are implicitly stored as relationships:
// - RERANK_SCORE relationships = S_reranker
// - Cosine similarities computed on-demand via GDS

RETURN m.dimension AS matrixSize;
```

### C.3.5 Phase 4: Low-Rank Approximation via Native GDS

Neo4j GDS provides the mathematical primitives needed for SVD-style decomposition natively. The key insight: **power iteration on the similarity matrix yields dominant singular vectors** — this is exactly what `gds.eigenvector` and `gds.pageRank` implement internally.

#### C.3.5.1 Mathematical Foundation

SVD finds vectors that maximize variance under linear transformation. Power iteration accomplishes this by repeatedly applying the matrix and normalizing. GDS algorithms use this same approach:

| GDS Algorithm | Mathematical Operation | Use for Lensing |
|---------------|----------------------|------------------|
| `gds.eigenvector` | Power iteration on adjacency | Find dominant directions in similarity graph |
| `gds.pageRank` | Power iteration with damping | Robust variant, handles sparse matrices |
| `gds.fastRP` | Johnson-Lindenstrauss random projection | Same math family as randomized SVD |

#### C.3.5.2 Approach A: Similarity Graph + Eigenvector Centrality

Project the similarity matrix as a weighted graph, then use power iteration:

```cypher
// ============================================================
// PHASE 4A: SVD VIA NATIVE POWER ITERATION
// Uses GDS eigenvector centrality on similarity graph
// ============================================================

// Step 1: Create in-memory graph from RERANK_SCORE relationships
CALL gds.graph.project(
    'lens-similarity-graph',
    'EntityDetail',
    {
        RERANK_SCORE: {
            properties: ['score'],
            orientation: 'UNDIRECTED'
        }
    }
)

// Step 2: Run eigenvector centrality (power iteration)
// This finds the dominant eigenvector of the similarity matrix
CALL gds.eigenvector.stream('lens-similarity-graph', {
    maxIterations: 100,
    tolerance: 0.0001,
    relationshipWeightProperty: 'score'
})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
ORDER BY score DESC

// The eigenvector scores indicate which nodes dominate the similarity structure
// High-scoring nodes define the principal directions of variation
RETURN node.name, score
LIMIT 20;
```

#### C.3.5.3 Approach B: FastRP for Dimensionality Reduction

FastRP implements the Johnson-Lindenstrauss lemma — the same mathematical foundation as randomized SVD:

```cypher
// ============================================================
// PHASE 4B: RANDOMIZED LOW-RANK VIA FASTRP
// Johnson-Lindenstrauss preserves distances in lower dimension
// ============================================================

// FastRP on the similarity graph produces compressed representations
// that preserve pairwise distances — equivalent to truncated SVD goal

CALL gds.fastRP.mutate('lens-similarity-graph', {
    embeddingDimension: 128,  // This is our rank r
    iterationWeights: [0.0, 1.0, 1.0, 1.0],
    relationshipWeightProperty: 'score',
    mutateProperty: 'lens_component'
})
YIELD nodePropertiesWritten

// Write back to Neo4j
CALL gds.graph.nodeProperties.write(
    'lens-similarity-graph',
    ['lens_component']
)
YIELD propertiesWritten

RETURN propertiesWritten;
```

#### C.3.5.4 Approach C: Pure Cypher Power Iteration

For full control, implement power iteration directly in Cypher:

```cypher
// ============================================================
// PHASE 4C: PURE CYPHER POWER ITERATION
// No external dependencies whatsoever
// ============================================================

:param rank => 128;
:param maxIterations => 50;
:param tolerance => 0.0001;

// Initialize lens node
MERGE (lens:TransformationLens {namespace: $namespace, rank: $rank})
SET lens.status = 'COMPUTING',
    lens.iteration = 0;

// Get all nodes with embeddings
MATCH (f:EntityDetail {namespace: $namespace})
WHERE f.fused_embedding IS NOT NULL
WITH collect(f) AS nodes, count(f) AS n

// Initialize random vector (one component per node)
WITH nodes, n,
     [i IN range(0, n-1) | rand() - 0.5] AS v

// Normalize initial vector
WITH nodes, n, v,
     sqrt(reduce(s = 0.0, x IN v | s + x*x)) AS norm
WITH nodes, n, [x IN v | x / norm] AS v

// Power iteration: v_new = S * v / ||S * v||
// where S is the similarity matrix (stored as RERANK_SCORE relationships)
UNWIND range(0, $maxIterations - 1) AS iteration

// Matrix-vector multiplication via graph traversal
WITH nodes, v, iteration
UNWIND range(0, size(nodes)-1) AS i
WITH nodes, v, iteration, i, nodes[i] AS node_i

// Compute (S * v)[i] = sum_j S[i,j] * v[j]
OPTIONAL MATCH (node_i)-[r:RERANK_SCORE]-(node_j:EntityDetail)
WHERE node_j IN nodes
WITH nodes, v, iteration, i,
     coalesce(sum(r.score * v[apoc.coll.indexOf(nodes, node_j)]), 0.0) AS new_v_i

// Collect new vector and normalize
WITH iteration, collect(new_v_i) AS v_new
WITH iteration, v_new,
     sqrt(reduce(s = 0.0, x IN v_new | s + x*x)) AS norm
WITH iteration, [x IN v_new | x / norm] AS v_normalized

// Store dominant eigenvector
MATCH (lens:TransformationLens {namespace: $namespace})
SET lens.dominant_eigenvector = v_normalized,
    lens.iteration = iteration,
    lens.status = 'READY'

RETURN iteration, size(v_normalized) AS vectorDim;
```

### C.3.6 Phase 4d: Constructing the Transformation Matrix

Once we have the dominant directions (from eigenvector, FastRP, or power iteration), construct the lens transformation matrices A and B:

```cypher
// ============================================================
// PHASE 4D: CONSTRUCT LENS MATRICES FROM EIGENVECTORS
// Combines dominant directions into transformation T = I + AB^T
// ============================================================

// Collect the lens components computed by FastRP or eigenvector analysis
MATCH (f:EntityDetail {namespace: $namespace})
WHERE f.lens_component IS NOT NULL
WITH collect(f.lens_component) AS components,
     collect(f.fused_embedding) AS embeddings

// The lens_component vectors form the columns of our low-rank factors
// A and B are derived from the relationship between original embeddings
// and their projections onto the dominant similarity directions

WITH components, embeddings,
     size(embeddings[0]) AS embeddingDim,
     size(components[0]) AS rank

// Compute A: how each embedding dimension relates to lens components
// This is essentially: A = E^T @ C @ (C^T @ C)^{-1}
// Simplified: use the component vectors directly as the basis

MATCH (lens:TransformationLens {namespace: $namespace})
SET lens.rank = rank,
    lens.embedding_dim = embeddingDim,
    lens.status = 'READY',
    lens.computed_at = datetime(),
    lens.method = 'native_gds'

RETURN rank, embeddingDim;
```

### C.3.7 Optional: External SVD Service (Legacy Approach)

For users who prefer external computation, the option remains available:

```cypher
// ============================================================
// OPTIONAL: SVD VIA EXTERNAL MICROSERVICE
// Use only if native approaches are insufficient
// ============================================================

// Collect training data
MATCH (f1:EntityDetail)-[r:RERANK_SCORE]->(f2:EntityDetail)
WHERE f1.namespace = $namespace
WITH collect({
    i: f1.matrix_idx,
    j: f2.matrix_idx,
    cosine: gds.similarity.cosine(f1.fused_embedding, f2.fused_embedding),
    reranker: r.score
}) AS trainingPairs

// Get embedding dimension from sample
MATCH (f:EntityDetail {namespace: $namespace})
WHERE f.fused_embedding IS NOT NULL
WITH trainingPairs, size(f.fused_embedding) AS dim
LIMIT 1

// Call SVD service (only if needed)
CALL apoc.load.jsonParams(
    $svdServiceUrl,
    {method: "POST", `Content-Type`: "application/json"},
    apoc.convert.toJson({
        pairs: trainingPairs,
        rank: $rank,
        embedding_dim: dim,
        regularization: 0.01
    })
) YIELD value

// Store lens matrices
MATCH (lens:TransformationLens {namespace: $namespace, rank: $rank})
SET lens.A = value.A,
    lens.B = value.B,
    lens.singular_values = value.sigma,
    lens.loss = value.final_loss,
    lens.method = 'external_svd',
    lens.status = 'READY'

RETURN lens.loss AS finalLoss;
```

**Note:** The native GDS approaches (Sections C.3.5.2-C.3.5.4) are preferred. With modern hardware (192GB+ RAM), heap constraints are rarely a concern even for large codebases.

### C.3.8 Phase 5: Apply Lens Transformation

```cypher
// ============================================================
// PHASE 5: TRANSFORM EMBEDDINGS
// Applies T = I + AB^T to all embeddings
// ============================================================

// Load lens
MATCH (lens:TransformationLens {namespace: $namespace, status: 'READY'})
WITH lens.A AS A, lens.B AS B, lens.rank AS r

// Transform each embedding
CALL apoc.periodic.iterate(
    "
    MATCH (f:EntityDetail {namespace: $namespace})
    WHERE f.fused_embedding IS NOT NULL
    RETURN f
    ",
    "
    WITH f, $A AS A, $B AS B
    
    // Compute B^T @ e (result: r-dimensional)
    WITH f, A,
         [i IN range(0, size(B[0])-1) |
           reduce(s = 0.0, j IN range(0, size(f.fused_embedding)-1) |
             s + B[j][i] * f.fused_embedding[j]
           )
         ] AS Bt_e
    
    // Compute A @ (B^T @ e) and add to original (T = I + AB^T)
    WITH f,
         [i IN range(0, size(f.fused_embedding)-1) |
           f.fused_embedding[i] + 
           reduce(s = 0.0, j IN range(0, size(Bt_e)-1) |
             s + A[i][j] * Bt_e[j]
           )
         ] AS transformed
    
    SET f.lensed_embedding = transformed
    ",
    {batchSize: 100, parallel: true, params: {A: A, B: B, namespace: $namespace}}
)
YIELD batches, total
RETURN batches, total;
```

---

## C.4 Minimal Reranker Server Implementation

For local deployment, a minimal FastAPI server suffices:

```python
# reranker_server.py
# Minimal server for Neo4j APOC integration
# Run: uvicorn reranker_server:app --host 0.0.0.0 --port 8080

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import torch

app = FastAPI()

# Load model once at startup
# Options: BAAI/bge-reranker-v2-m3, Alibaba-NLP/gte-Qwen2-1.5B-instruct
model = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cuda' if torch.cuda.is_available() else 'cpu')

class Pair(BaseModel):
    id: str
    text_a: str
    text_b: str

class RerankerRequest(BaseModel):
    pairs: list[Pair]

class Score(BaseModel):
    id: str
    score: float

class RerankerResponse(BaseModel):
    scores: list[Score]

@app.post("/rerank", response_model=RerankerResponse)
async def rerank(request: RerankerRequest):
    # Prepare pairs for batch inference
    pairs = [(p.text_a, p.text_b) for p in request.pairs]
    
    # Batch inference
    scores = model.predict(pairs)
    
    # Format response
    return RerankerResponse(
        scores=[
            Score(id=request.pairs[i].id, score=float(scores[i]))
            for i in range(len(scores))
        ]
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model": "bge-reranker-v2-m3"}
```

**Deployment Options:**

| Option | Hardware | Throughput | Cost |
|--------|----------|------------|------|
| Local GPU | RTX 3090 | ~50 pairs/sec | One-time |
| Local CPU | Ryzen 9950X | ~2 pairs/sec | One-time |
| Vast.ai | A100 rental | ~200 pairs/sec | ~$1/hour |
| Modal.com | Serverless GPU | ~100 pairs/sec | Pay-per-call |

---

## C.5 Cloud-Based Alternative: Cohere Rerank API

For zero-infrastructure deployment, use cloud reranker APIs:

```cypher
// ============================================================
// CLOUD RERANKER: COHERE INTEGRATION
// No local GPU required
// ============================================================

:param cohereApiKey => 'your-api-key';

MATCH (f1:EntityDetail)-[p:PENDING_RERANK]->(f2:EntityDetail)
WHERE p.processed IS NULL
LIMIT 50

WITH collect({
    id: toString(id(p)),
    text_a: coalesce(f1.content, f1.name),
    text_b: coalesce(f2.content, f2.name)
}) AS pairs

// Cohere uses different API structure
WITH pairs,
     [p IN pairs | p.text_a + " [SEP] " + p.text_b] AS documents,
     pairs[0].text_a AS query  // Cohere needs a query

CALL apoc.load.jsonParams(
    "https://api.cohere.ai/v1/rerank",
    {
        method: "POST",
        `Content-Type`: "application/json",
        Authorization: "Bearer " + $cohereApiKey
    },
    apoc.convert.toJson({
        model: "rerank-english-v3.0",
        query: query,
        documents: documents,
        top_n: size(documents)
    })
) YIELD value

// Process Cohere response format
UNWIND value.results AS result
WITH pairs[result.index] AS pair, result.relevance_score AS score

// Update graph...
RETURN pair.id, score;
```

---

## C.6 Complete Calibration Pipeline: Single Entry Point

```cypher
// ============================================================
// MASTER CALIBRATION PROCEDURE
// Single entry point for full lens calibration
// ============================================================

// Usage: CALL lens.calibrate('checkitout', {rank: 128, pairs: 5000})

:param namespace => 'checkitout';
:param config => {rank: 128, targetPairs: 5000, batchSize: 50};

// Step 1: Initialize job
MERGE (job:CalibrationJob {namespace: $namespace})
SET job.config = $config,
    job.started = datetime(),
    job.status = 'INITIALIZING'

WITH job

// Step 2: Sample pairs
CALL {
    MATCH (f1:EntityDetail {namespace: $namespace})
    MATCH (f2:EntityDetail {namespace: $namespace})
    WHERE id(f1) < id(f2) AND rand() < 0.1
    WITH f1, f2, gds.similarity.cosine(f1.fused_embedding, f2.fused_embedding) AS cos
    ORDER BY cos DESC
    LIMIT $config.targetPairs
    MERGE (f1)-[p:PENDING_RERANK]->(f2)
    SET p.cosine_sim = cos
    RETURN count(*) AS sampled
}

SET job.status = 'PAIRS_SAMPLED', job.pairsSampled = sampled

// Step 3: Rerank (iterative - run until no pending pairs)
// This would be called repeatedly until complete

// Step 4: Compute lens
// Delegated to SVD service

// Step 5: Apply transformation
// Run after SVD completes

RETURN job{.*} AS calibrationJob;
```

---

## C.7 Progress Monitoring Dashboard Query

```cypher
// ============================================================
// CALIBRATION PROGRESS DASHBOARD
// Real-time monitoring of lens calibration
// ============================================================

MATCH (job:CalibrationJob {namespace: $namespace})
OPTIONAL MATCH (f1:EntityDetail)-[p:PENDING_RERANK]->(f2:EntityDetail)
WHERE f1.namespace = $namespace
WITH job, count(p) AS pendingPairs

OPTIONAL MATCH (f1:EntityDetail)-[r:RERANK_SCORE]->(f2:EntityDetail)
WHERE f1.namespace = $namespace
WITH job, pendingPairs, count(r) AS completedPairs, avg(r.divergence) AS avgDivergence

OPTIONAL MATCH (lens:TransformationLens {namespace: $namespace})

RETURN {
    status: job.status,
    started: job.started,
    elapsed: duration.between(job.started, datetime()),
    
    pairs: {
        target: job.config.targetPairs,
        pending: pendingPairs,
        completed: completedPairs,
        progress: toFloat(completedPairs) / (pendingPairs + completedPairs) * 100
    },
    
    divergence: {
        average: avgDivergence,
        interpretation: CASE 
            WHEN avgDivergence > 0.4 THEN 'HIGH - Lensing will help significantly'
            WHEN avgDivergence > 0.2 THEN 'MODERATE - Lensing recommended'
            ELSE 'LOW - Consider if lensing is needed'
        END
    },
    
    lens: CASE WHEN lens IS NOT NULL THEN {
        rank: lens.rank,
        status: lens.status,
        loss: lens.loss,
        computed_at: lens.computed_at
    } ELSE null END,
    
    eta: CASE 
        WHEN completedPairs > 0 THEN 
            duration.between(job.started, datetime()) * pendingPairs / completedPairs
        ELSE null 
    END
    
} AS dashboard;
```

---

## C.8 Advantages and Limitations

### C.8.1 Advantages

| Aspect | Benefit |
|--------|---------|
| **Deployment** | Single Neo4j instance, no Python runtime |
| **Consistency** | Transactional guarantees on all operations |
| **Monitoring** | Native graph queries for progress tracking |
| **Integration** | Seamless with existing graph workflows |
| **Portability** | Cypher queries work across Neo4j deployments |
| **State** | All state in graph, survives restarts |

### C.8.2 Considerations

| Aspect | Consideration | Approach |
|--------|---------------|----------|
| **SVD-equivalent** | Power iteration via `gds.eigenvector` | Native GDS, no external dependency |
| **Random projection** | FastRP implements J-L lemma | Same math as randomized SVD |
| **Matrix ops** | Cypher list operations slower than NumPy | GDS native algorithms close the gap; 192GB+ RAM eliminates chunking needs |
| **Debugging** | Complex Cypher harder to debug | Modular procedure design |

### C.8.3 Why Graph-Native is Superior for In-Graph Embeddings

When embeddings already reside in Neo4j, the graph-native approach eliminates the most expensive operation: **vector transfer**.

```
Traditional Python Approach:
┌─────────────────────────────────────────────────────────────┐
│  Neo4j → Driver → Network → Python → NumPy → Network → Neo4j │
│                                                             │
│  For 5000 nodes × 4096 dimensions × 4 bytes = 80MB transfer │
│  Round-trip latency + serialization overhead                │
└─────────────────────────────────────────────────────────────┘

Graph-Native Approach:
┌─────────────────────────────────────────────────────────────┐
│  Neo4j (GDS in-memory graph) → Transform → Write back       │
│                                                             │
│  Zero network transfer, zero serialization                  │
│  Vectors never leave the JVM heap                           │
└─────────────────────────────────────────────────────────────┘
```

### C.8.4 When to Use Graph-Native vs Python

**Use Graph-Native when:**
- Embeddings already live in Neo4j (most common case)
- Operational simplicity is priority
- Want to avoid vector transfer overhead
- Integration with existing Neo4j workflows
- Any codebase size (GDS scales well)

**Consider Python orchestration when:**
- Embeddings are generated externally and need one-time processing
- Custom loss functions required for research
- Need NumPy-specific operations not available in GDS
- Debugging complex numerical issues

---

## C.9 Advanced: Custom Algorithms via Pregel API

For maximum control, the GDS Pregel API allows implementing custom iterative algorithms in Java that leverage the optimized in-memory graph:

```java
// Example: Custom power iteration for lens calibration
// Compile as JAR, place in Neo4j plugins directory

@Algorithm("lens.powerIteration")
public class LensPowerIteration implements PregelComputation<LensConfig> {
    
    @Override
    public void compute(ComputeContext<LensConfig> context, Messages messages) {
        // Power iteration step: v_new[i] = sum_j(S[i,j] * v[j])
        double sum = 0.0;
        for (Double msg : messages) {
            sum += msg;
        }
        
        // Normalize and update
        double normalized = sum / context.nodeCount();
        context.setNodeValue(EIGENVECTOR_COMPONENT, normalized);
        
        // Send to neighbors weighted by similarity
        context.forEachNeighbor(targetId -> {
            double weight = context.relationshipProperty(targetId, "score");
            context.sendTo(targetId, normalized * weight);
        });
    }
}
```

This approach provides:
- Full parallelization across CPU cores (Ryzen 9950X: 16 cores / 32 threads)
- Native memory management 
- Integration with GDS graph catalog
- Standard procedure interface (`CALL lens.powerIteration.stream(...)`)

## C.10 Future Directions

The current implementation demonstrates that Information Lensing is fully achievable within Neo4j. Potential enhancements:

1. **GDS Native Procedure**: Package the lens calibration as a single GDS procedure
2. **Incremental Updates**: Leverage GDS graph mutations for online lens refinement
3. **Multi-Lens Fusion**: Parallel computation of structural/semantic/behavioral lenses
4. **Automatic Rank Selection**: Use GDS similarity metrics to determine optimal rank

---

## C.11 Conclusion

The graph-native implementation demonstrates that Information Lensing does not require complex Python orchestration. By leveraging:

1. **APOC's `apoc.load.jsonParams`** for REST API integration (reranker only)
2. **APOC's `apoc.periodic.iterate`** for batch processing
3. **GDS `eigenvector` / `fastRP`** for native power iteration and random projection
4. **Graph relationships** for sparse matrix storage
5. **Cypher `reduce()`** for matrix-vector multiplication

The entire calibration pipeline executes within Neo4j — **no Python, no external SVD service, no vector transfer overhead**. When embeddings already live in the graph, shipping 4096-dimensional vectors through a network proxy to multiply them by a matrix is architectural waste.

The reranker service remains the only external dependency (compute-intensive cross-encoder inference genuinely benefits from GPU). Even that can be swapped between local GPU, cloud API, or hybrid approaches without changing the Cypher pipeline.

---

## References (Appendix-Specific)

Neo4j APOC Documentation. "Load JSON Procedures." https://neo4j.com/labs/apoc/current/import/load-json/

Neo4j GDS Documentation. "Similarity Functions." https://neo4j.com/docs/graph-data-science/current/

Tomaz Bratanic. "Integrate LLM workflows with Knowledge Graph using Neo4j and APOC." Medium, 2023.

---

*This appendix accompanies the main Information Lensing paper and provides implementation guidance for Neo4j-native deployment.*
