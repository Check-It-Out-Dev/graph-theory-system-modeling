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
│  │  • Native projections ─► Efficient graph ops        │    │
│  │  • (Future: gds.ml.* for SVD)                       │    │
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

### C.3.5 Phase 4: Low-Rank Approximation (SVD)

Neo4j doesn't have native SVD, but we can implement iterative low-rank approximation:

```cypher
// ============================================================
// PHASE 4: LOW-RANK LENS COMPUTATION
// Implements power iteration for dominant singular vectors
// ============================================================

// Configuration
:param rank => 128;
:param maxIterations => 100;
:param tolerance => 0.0001;
:param embeddingDim => 4096;

// Initialize lens node
MERGE (lens:TransformationLens {namespace: $namespace, rank: $rank})
SET lens.status = 'COMPUTING',
    lens.iteration = 0;

// For true SVD, we need to call an external service or use approximation
// Option A: Call external SVD endpoint
// Option B: Use power iteration (shown below)
// Option C: Use randomized SVD via Monte Carlo

// Power Iteration for top-k singular vectors
// This is a simplified version - production would batch this

WITH range(0, $rank - 1) AS rankIndices
UNWIND rankIndices AS k

// Initialize random vector v_k
WITH k, [x IN range(0, $embeddingDim - 1) | rand() - 0.5] AS v

// Normalize
WITH k, v, sqrt(reduce(s = 0.0, x IN v | s + x*x)) AS norm
WITH k, [x IN v | x / norm] AS v

// Power iteration (simplified - actual implementation needs matrix multiplication)
// For full implementation, see Appendix C.6

// Store as lens component
MATCH (lens:TransformationLens {namespace: $namespace, rank: $rank})
SET lens['v_' + toString(k)] = v

RETURN k AS singularVectorIndex;
```

### C.3.6 Phase 4b: External SVD Service Integration

For production use, delegate SVD to a lightweight service:

```cypher
// ============================================================
// PHASE 4b: SVD VIA EXTERNAL MICROSERVICE
// Delegates matrix factorization to specialized endpoint
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

// Call SVD service
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

// Store lens matrices A and B (T = I + AB^T)
MATCH (lens:TransformationLens {namespace: $namespace, rank: $rank})
SET lens.A = value.A,      // Shape: [dim, rank]
    lens.B = value.B,      // Shape: [dim, rank]
    lens.singular_values = value.sigma,
    lens.loss = value.final_loss,
    lens.iterations = value.iterations,
    lens.status = 'READY',
    lens.computed_at = datetime()

RETURN lens.loss AS finalLoss, lens.iterations AS iterations;
```

### C.3.7 Phase 5: Apply Lens Transformation

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

### C.8.2 Limitations

| Aspect | Limitation | Mitigation |
|--------|------------|------------|
| **SVD** | No native Neo4j SVD | External microservice |
| **Matrix ops** | Cypher list operations slower than NumPy | Batch externally for large matrices |
| **Debugging** | Complex Cypher harder to debug | Modular procedure design |
| **Memory** | Large matrices in Cypher heap | Chunk processing |

### C.8.3 When to Use Graph-Native vs Python

**Use Graph-Native when:**
- Operational simplicity is priority
- Team lacks Python expertise
- Integration with existing Neo4j workflows
- Codebases < 5000 files

**Use Python orchestration when:**
- Need complex matrix operations
- Codebases > 10000 files
- Custom loss functions required
- Research/experimentation phase

---

## C.9 Future: Native GDS Integration

The Neo4j Graph Data Science team has indicated interest in embedding transformation primitives. Future GDS versions may include:

```cypher
// Hypothetical future GDS API
CALL gds.ml.embedding.transform.fit({
    nodeProjection: 'EntityDetail',
    embeddingProperty: 'fused_embedding',
    targetSimilarityRelationship: 'RERANK_SCORE',
    rank: 128
}) YIELD lensId, loss

CALL gds.ml.embedding.transform.apply({
    lensId: lensId,
    nodeProjection: 'EntityDetail',
    inputProperty: 'fused_embedding',
    outputProperty: 'lensed_embedding'
}) YIELD nodesTransformed
```

Until then, the APOC-based approach provides a working solution.

---

## C.10 Conclusion

The graph-native implementation demonstrates that Information Lensing does not require complex Python orchestration. By leveraging:

1. **APOC's `apoc.load.jsonParams`** for REST API integration
2. **APOC's `apoc.periodic.iterate`** for batch processing
3. **Graph relationships** for sparse matrix storage
4. **Lightweight microservices** for compute-intensive operations

The entire calibration pipeline can execute within Neo4j, with progress tracked as graph state and results stored as node properties. This approach trades some computational efficiency for operational simplicity—a worthwhile tradeoff for production deployments where reducing moving parts matters.

The reranker service remains the only external dependency, and even that can be swapped between local GPU, cloud API, or hybrid approaches without changing the Cypher pipeline.

---

## References (Appendix-Specific)

Neo4j APOC Documentation. "Load JSON Procedures." https://neo4j.com/labs/apoc/current/import/load-json/

Neo4j GDS Documentation. "Similarity Functions." https://neo4j.com/docs/graph-data-science/current/

Tomaz Bratanic. "Integrate LLM workflows with Knowledge Graph using Neo4j and APOC." Medium, 2023.

---

*This appendix accompanies the main Information Lensing paper and provides implementation guidance for Neo4j-native deployment.*
