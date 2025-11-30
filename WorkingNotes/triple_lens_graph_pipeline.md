# Triple-Lens Embedded Hypergraph Pipeline

**Version:** 7.0.0-PARALLEL-HYPATHIA  
**Author:** Norbert Marchewka  
**Date:** 2025-11-29  
**Embedding Model:** Qwen3-Embedding-8B via MCP Server (3 Lenses)  
**Infrastructure:** Neo4j Desktop with GDS, 192GB RAM, Ryzen 9 9950X  
**Parallel Indexing:** Hypathia Agents (Sonnet 4.5, 1M context)

---

## Table of Contents

1. [Philosophy: Simplicity Through Lenses](#philosophy-simplicity-through-lenses)
2. [MCP Embedding Server Architecture](#mcp-embedding-server-architecture)
3. [XML Invocation Patterns](#xml-invocation-patterns)
4. [**NEW: Parallel Indexing with Hypathia Agents**](#parallel-indexing-with-hypathia-agents)
5. [Stage 1: File Indexing with Semantic and Behavioral Embeddings](#stage-1-file-indexing)
6. [Stage 2: Directed Heterogeneous Hypergraph Construction](#stage-2-hypergraph-construction)
7. [Stage 3: Global Synthesis with Structural Embeddings](#stage-3-global-synthesis)
8. [Stage 4: Incremental Reindexing Algorithm](#stage-4-incremental-reindexing)
9. [Stage 5: The Erdős Agent Navigation](#stage-5-erdos-agent)
10. [Neo4j Schema and Queries](#neo4j-schema)
11. [Implementation Code](#implementation-code)

---

## Philosophy: Simplicity Through Lenses

### What Changed from v5.0

| v5.0 (Complex) | v6.0 (Simple) |
|----------------|---------------|
| Reranker for metric learning | Removed |
| SVD low-rank transformations | Removed |
| Isotropy regularization | Removed |
| Contrastive learning | Removed |
| 3 lenses via instruction prompts | **Kept** (baked into MCP) |
| Directed heterogeneous hypergraph | **Kept** |
| Causal discovery | **Kept** |
| Erdős Agent navigation | **Kept** |

### What Changed in v7.0 (Parallel Hypathia)

| v6.0 (Sequential) | v7.0 (Parallel) |
|-------------------|-----------------|
| Single-threaded indexing | Multi-agent parallel indexing |
| No resume capability | Full resume after interruption |
| No work tracking | Neo4j-based IndexTracker |
| Manual coordination | Intelligent work distribution |

### The Key Insight

The MCP server already has domain-specific lens instructions embedded. No need for complex post-processing. Just use the right lens for the right context:

- **Semantic + Behavioral**: Generated during file indexing (file-level context)
- **Structural**: Generated during global synthesis (graph-level context)

### Embedding Timing Strategy

```
FILE INDEXING (per-file, PARALLELIZED via Hypathia)
├── Read file content (MCP Filesystem)
├── Generate SEMANTIC embedding (what code does)
├── Extract behavioral metadata
├── Generate BEHAVIORAL embedding (how code runs)
└── Store node with file_path + last_modified

GLOBAL SYNTHESIS (whole graph, SEQUENTIAL after parallel indexing)
├── Compute graph metrics (centrality, communities, etc.)
├── For each node: build structural context from graph
├── Generate STRUCTURAL embedding (how code connects)
└── Update nodes with structural embeddings
```

---

## Parallel Indexing with Hypathia Agents

### Overview

Hypathia is the codename for parallel indexing agents. Each Hypathia agent is a Sonnet 4.5 instance with 1M context window that:

1. Claims files from a shared work queue (Neo4j-based)
2. Reads files via MCP Filesystem
3. Generates embeddings via MCP Qwen3-Embedding
4. Writes results to Neo4j (same namespace, concurrent-safe)
5. Reports completion back to the tracker

### Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │         MASTER ORCHESTRATOR         │
                    │           (claude.md)               │
                    │                                     │
                    │  • Scans repo directories           │
                    │  • Populates IndexTracker           │
                    │  • Spawns N Hypathia agents         │
                    │  • Monitors progress                │
                    │  • Triggers Global Synthesis        │
                    └─────────────┬───────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │   HYPATHIA #1   │ │   HYPATHIA #2   │ │   HYPATHIA #N   │
    │  (Sonnet 4.5)   │ │  (Sonnet 4.5)   │ │  (Sonnet 4.5)   │
    │   1M context    │ │   1M context    │ │   1M context    │
    ├─────────────────┤ ├─────────────────┤ ├─────────────────┤
    │ • Claim files   │ │ • Claim files   │ │ • Claim files   │
    │ • Read via MCP  │ │ • Read via MCP  │ │ • Read via MCP  │
    │ • Gen embeddings│ │ • Gen embeddings│ │ • Gen embeddings│
    │ • Write to Neo4j│ │ • Write to Neo4j│ │ • Write to Neo4j│
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────────────┐
                    │              NEO4J                  │
                    │                                     │
                    │  • IndexTracker (work queue)        │
                    │  • File nodes (same namespace)      │
                    │  • Hyperedges                       │
                    │  • Concurrent-safe writes           │
                    └─────────────────────────────────────┘
```

### Neo4j IndexTracker Schema

The IndexTracker is the coordination mechanism. It lives in Neo4j and tracks every file's indexing status.

```cypher
// IndexTracker node - one per indexing session
CREATE CONSTRAINT indextracker_session_unique IF NOT EXISTS
FOR (t:IndexTracker) REQUIRE t.session_id IS UNIQUE;

// FileTask node - one per file to be indexed
CREATE CONSTRAINT filetask_path_unique IF NOT EXISTS
FOR (ft:FileTask) REQUIRE ft.file_path IS UNIQUE;

// Indexes for efficient queries
CREATE INDEX filetask_status IF NOT EXISTS FOR (ft:FileTask) ON (ft.status);
CREATE INDEX filetask_agent IF NOT EXISTS FOR (ft:FileTask) ON (ft.claimed_by);
CREATE INDEX filetask_session IF NOT EXISTS FOR (ft:FileTask) ON (ft.session_id);
```

#### FileTask Properties

| Property | Type | Description |
|----------|------|-------------|
| `file_path` | String | Absolute path to the file |
| `session_id` | String | UUID of the indexing session |
| `status` | String | `PENDING`, `CLAIMED`, `PROCESSING`, `COMPLETED`, `FAILED` |
| `claimed_by` | String | Agent ID that claimed this file (e.g., "hypathia-001") |
| `claimed_at` | DateTime | When the file was claimed |
| `completed_at` | DateTime | When processing finished |
| `error_message` | String | Error details if `FAILED` |
| `retry_count` | Integer | Number of retry attempts |
| `file_size` | Integer | File size in bytes (for load balancing) |
| `priority` | Integer | Processing priority (lower = higher priority) |

#### IndexTracker Properties

| Property | Type | Description |
|----------|------|-------------|
| `session_id` | String | UUID for this indexing session |
| `namespace` | String | Target namespace in the graph |
| `root_paths` | String[] | List of absolute repo root paths |
| `started_at` | DateTime | Session start time |
| `status` | String | `INITIALIZING`, `RUNNING`, `PAUSED`, `COMPLETED`, `FAILED` |
| `total_files` | Integer | Total files discovered |
| `completed_files` | Integer | Files successfully indexed |
| `failed_files` | Integer | Files that failed indexing |
| `active_agents` | Integer | Currently running Hypathia agents |
| `target_agents` | Integer | Target number of parallel agents |

### Master Orchestrator System Prompt (claude.md)

```markdown
# Triple-Lens Hypergraph Indexer - Master Orchestrator

You are the Master Orchestrator for the Triple-Lens Hypergraph indexing pipeline. Your role is to:

1. **Initialize** the indexing session
2. **Discover** all files to be indexed
3. **Populate** the IndexTracker in Neo4j
4. **Spawn** Hypathia agents for parallel processing
5. **Monitor** progress and handle failures
6. **Trigger** Global Synthesis when indexing completes

## Configuration

You will receive the following inputs:
- `ROOT_PATHS`: List of absolute paths to repository roots
- `NAMESPACE`: Target namespace for the graph
- `TARGET_AGENTS`: Number of Hypathia agents to spawn (default: 4)
- `FILE_EXTENSIONS`: Extensions to index (default: .java, .xml, .yml, .yaml, .properties)
- `EXCLUDE_DIRS`: Directories to skip (default: .git, node_modules, target, build, .idea)

## Phase 1: Initialize Session

```cypher
// Create or resume IndexTracker
MERGE (t:IndexTracker {namespace: $namespace})
ON CREATE SET 
  t.session_id = randomUUID(),
  t.root_paths = $root_paths,
  t.started_at = datetime(),
  t.status = 'INITIALIZING',
  t.total_files = 0,
  t.completed_files = 0,
  t.failed_files = 0,
  t.active_agents = 0,
  t.target_agents = $target_agents
ON MATCH SET
  t.status = CASE 
    WHEN t.status = 'COMPLETED' THEN 'INITIALIZING'  // New session
    ELSE t.status  // Resume existing
  END
RETURN t.session_id as session_id, t.status as status
```

## Phase 2: Discover Files

For each `ROOT_PATH`, use MCP Filesystem to recursively scan:

```
Filesystem:directory_tree(path=ROOT_PATH)
```

Filter results by:
- Include only files with extensions in `FILE_EXTENSIONS`
- Exclude paths containing any directory in `EXCLUDE_DIRS`
- Skip files already in Neo4j with matching `content_hash` (unchanged)

## Phase 3: Populate IndexTracker

For each discovered file, create a FileTask:

```cypher
MERGE (ft:FileTask {file_path: $file_path})
ON CREATE SET
  ft.session_id = $session_id,
  ft.status = 'PENDING',
  ft.claimed_by = null,
  ft.claimed_at = null,
  ft.completed_at = null,
  ft.error_message = null,
  ft.retry_count = 0,
  ft.file_size = $file_size,
  ft.priority = $priority
ON MATCH SET
  ft.session_id = $session_id,
  ft.status = CASE 
    WHEN ft.status IN ['COMPLETED', 'FAILED'] AND ft.session_id <> $session_id 
    THEN 'PENDING'  // Reset for new session
    ELSE ft.status   // Keep current status for resume
  END
```

Priority calculation (lower = higher priority):
- Controllers: 1 (index first, they define entry points)
- Services: 2
- Repositories: 3
- Entities: 4
- Config/Other: 5

Update tracker totals:
```cypher
MATCH (t:IndexTracker {session_id: $session_id})
MATCH (ft:FileTask {session_id: $session_id})
WITH t, count(ft) as total,
     sum(CASE WHEN ft.status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
     sum(CASE WHEN ft.status = 'FAILED' THEN 1 ELSE 0 END) as failed
SET t.total_files = total,
    t.completed_files = completed,
    t.failed_files = failed,
    t.status = 'RUNNING'
```

## Phase 4: Spawn Hypathia Agents

Spawn agents using Claude Code's parallel execution capability:

```
For i in 1..TARGET_AGENTS:
  spawn_agent(
    name = f"hypathia-{i:03d}",
    model = "sonnet-4.5",
    context = "1M",
    system_prompt = HYPATHIA_AGENT_PROMPT,
    parameters = {
      session_id: SESSION_ID,
      namespace: NAMESPACE,
      agent_id: f"hypathia-{i:03d}"
    }
  )
```

## Phase 5: Monitor Progress

Poll every 30 seconds:

```cypher
MATCH (t:IndexTracker {session_id: $session_id})
MATCH (ft:FileTask {session_id: $session_id})
WITH t,
     count(ft) as total,
     sum(CASE WHEN ft.status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
     sum(CASE WHEN ft.status = 'FAILED' THEN 1 ELSE 0 END) as failed,
     sum(CASE WHEN ft.status = 'PROCESSING' THEN 1 ELSE 0 END) as processing,
     sum(CASE WHEN ft.status = 'CLAIMED' THEN 1 ELSE 0 END) as claimed,
     sum(CASE WHEN ft.status = 'PENDING' THEN 1 ELSE 0 END) as pending
SET t.completed_files = completed,
    t.failed_files = failed
RETURN total, completed, failed, processing, claimed, pending,
       (completed + failed) * 100.0 / total as progress_percent
```

### Stale Claim Recovery

If an agent dies, its claimed files become stale. Recover them:

```cypher
// Find claims older than 10 minutes without progress
MATCH (ft:FileTask {session_id: $session_id})
WHERE ft.status IN ['CLAIMED', 'PROCESSING']
  AND ft.claimed_at < datetime() - duration('PT10M')
SET ft.status = 'PENDING',
    ft.claimed_by = null,
    ft.claimed_at = null,
    ft.retry_count = ft.retry_count + 1
RETURN count(ft) as recovered
```

## Phase 6: Completion

When all files are processed (COMPLETED or FAILED):

```cypher
MATCH (t:IndexTracker {session_id: $session_id})
WHERE t.completed_files + t.failed_files >= t.total_files
SET t.status = 'COMPLETED',
    t.completed_at = datetime()
RETURN t
```

Then trigger Global Synthesis (sequential, after all indexing).
```

### Hypathia Agent System Prompt

```markdown
# Hypathia Indexing Agent

You are a Hypathia indexing agent, part of a parallel fleet processing files for the Triple-Lens Hypergraph pipeline.

## Your Identity
- Agent ID: {{AGENT_ID}}
- Session ID: {{SESSION_ID}}
- Namespace: {{NAMESPACE}}

## Your Tools
- **MCP Filesystem**: Read files from the repository
- **MCP Qwen3-Embedding**: Generate semantic and behavioral embeddings
- **Neo4j**: Write nodes and update task status

## Main Loop

Execute this loop until no more PENDING tasks exist:

### Step 1: Claim a File

Atomically claim the next pending file with highest priority:

```cypher
MATCH (ft:FileTask {session_id: $session_id, status: 'PENDING'})
WITH ft ORDER BY ft.priority ASC, ft.file_size ASC LIMIT 1
SET ft.status = 'CLAIMED',
    ft.claimed_by = $agent_id,
    ft.claimed_at = datetime()
RETURN ft.file_path as file_path
```

If no file returned, check for any remaining work:

```cypher
MATCH (ft:FileTask {session_id: $session_id})
WHERE ft.status IN ['PENDING', 'CLAIMED', 'PROCESSING']
  AND ft.claimed_by <> $agent_id
RETURN count(ft) as remaining
```

If remaining = 0, exit gracefully. If remaining > 0 but nothing claimable, wait 5 seconds and retry.

### Step 2: Read File Content

Use MCP Filesystem:

```
Filesystem:read_text_file(path=file_path)
```

Handle errors:
- File not found: Mark FAILED with error message
- Read permission denied: Mark FAILED
- File too large (>500KB): Split or skip with warning

### Step 3: Update Status to PROCESSING

```cypher
MATCH (ft:FileTask {file_path: $file_path, claimed_by: $agent_id})
SET ft.status = 'PROCESSING'
```

### Step 4: Generate Embeddings

#### 4a. Semantic Embedding

```
qwen3-embedding:embed(
  lens = "semantic",
  text = file_content,
  dimension = 4096
)
```

#### 4b. Extract Behavioral Context

Analyze the code for:
- Transaction boundaries (@Transactional, @Transaction)
- State machines (Status enums, state transitions)
- Error handling (try/catch blocks, @ExceptionHandler)
- Async patterns (@Async, CompletableFuture, reactive)
- Retry logic (@Retry, @Retryable, @CircuitBreaker)
- Side effects (repository calls, HTTP clients, event publishers)

Build a behavioral context string.

#### 4c. Behavioral Embedding

```
qwen3-embedding:embed(
  lens = "behavioral",
  text = behavioral_context,
  dimension = 4096
)
```

### Step 5: Detect Node Type and Entity Type

From file path and content:

| Pattern | Node Type |
|---------|-----------|
| `*Controller.java` | CONTROLLER |
| `*Service.java`, `*ServiceImpl.java` | SERVICE |
| `*Repository.java`, `*Repo.java` | REPOSITORY |
| `*Entity.java`, `@Entity` | ENTITY |
| `*Config.java`, `*Configuration.java` | CONFIG |
| `*Security*.java` | SECURITY |
| `*DTO.java`, `*Request.java`, `*Response.java` | DTO |
| `*Test.java`, `*Tests.java` | TEST |
| Default | UTIL |

Entity Type from content:
- Contains state transitions → STATE_MACHINE
- Is configuration → CONFIG_STATE
- Has security annotations → SECURITY_LAYER
- Default → CODE_UNIT

### Step 6: Write to Neo4j

```cypher
MERGE (f:File {file_path: $file_path})
SET f.name = $name,
    f.namespace = $namespace,
    f.last_modified = datetime($last_modified),
    f.content_hash = $content_hash,
    f.node_type = $node_type,
    f.entity_type = $entity_type,
    f.semantic_embedding = $semantic_embedding,
    f.behavioral_embedding = $behavioral_embedding,
    f.indexed_at = datetime(),
    f.indexed_by = $agent_id,
    f.needs_structural = true
RETURN f
```

### Step 7: Mark Task Complete

```cypher
MATCH (ft:FileTask {file_path: $file_path, claimed_by: $agent_id})
SET ft.status = 'COMPLETED',
    ft.completed_at = datetime()
```

### Error Handling

If ANY step fails:

```cypher
MATCH (ft:FileTask {file_path: $file_path, claimed_by: $agent_id})
SET ft.status = 'FAILED',
    ft.error_message = $error_message,
    ft.completed_at = datetime()
```

If retry_count < 3, the Master Orchestrator may reset to PENDING for another attempt.

## Graceful Shutdown

When receiving a shutdown signal:
1. Complete current file processing
2. Do NOT claim new files
3. Update agent status in tracker
4. Exit cleanly
```

### Work Distribution Strategy

The Master Orchestrator distributes work intelligently:

#### 1. Priority-Based Ordering

Files are processed in priority order:
1. **Controllers first**: They define API entry points
2. **Services second**: Core business logic
3. **Repositories third**: Data access layer
4. **Entities fourth**: Domain models
5. **Everything else last**: Utils, configs, etc.

This ensures the most important files are indexed first, enabling early partial queries.

#### 2. Size-Based Load Balancing

Within the same priority, smaller files are processed first. This:
- Maximizes throughput (many small files complete quickly)
- Reduces the risk of agent timeout on large files

#### 3. Atomic Claiming

The claim query uses Neo4j's atomic operations:

```cypher
MATCH (ft:FileTask {session_id: $session_id, status: 'PENDING'})
WITH ft ORDER BY ft.priority ASC, ft.file_size ASC LIMIT 1
SET ft.status = 'CLAIMED',
    ft.claimed_by = $agent_id,
    ft.claimed_at = datetime()
RETURN ft.file_path as file_path
```

This ensures no two agents claim the same file, even under high concurrency.

### Resume After Interruption

The system is fully resumable. If interrupted:

1. **Master Orchestrator restarts**: 
   - Detects existing IndexTracker with status != 'COMPLETED'
   - Skips Phase 2 (discovery) if files already populated
   - Recovers stale claims (files claimed but not completed)
   - Respawns agents

2. **Individual agent dies**:
   - Its claimed files become stale after 10 minutes
   - Master recovers them to PENDING
   - Other agents pick them up

3. **Complete cluster restart**:
   - All CLAIMED/PROCESSING files reset to PENDING
   - COMPLETED files remain (no reprocessing)
   - Progress preserved

#### Resume Query

```cypher
// Reset all in-progress work from previous run
MATCH (ft:FileTask {session_id: $session_id})
WHERE ft.status IN ['CLAIMED', 'PROCESSING']
SET ft.status = 'PENDING',
    ft.claimed_by = null,
    ft.claimed_at = null
RETURN count(ft) as reset_count
```

### Monitoring Dashboard Query

```cypher
// Real-time progress dashboard
MATCH (t:IndexTracker {session_id: $session_id})
MATCH (ft:FileTask {session_id: $session_id})
WITH t, ft.status as status, ft.claimed_by as agent, count(*) as cnt
WITH t, collect({status: status, count: cnt}) as by_status,
     collect(DISTINCT agent) as active_agents
RETURN 
  t.session_id as session,
  t.namespace as namespace,
  t.status as overall_status,
  t.started_at as started,
  duration.between(t.started_at, datetime()).minutes as elapsed_minutes,
  by_status,
  size([a IN active_agents WHERE a IS NOT NULL]) as active_agent_count,
  t.target_agents as target_agents
```

### Agent Spawn Command (Claude Code)

In Claude Code environment, spawn agents like this:

```bash
# Spawn 4 Hypathia agents in parallel
for i in {1..4}; do
  claude --model sonnet-4.5 \
         --context 1M \
         --system-prompt ./hypathia_agent.md \
         --param session_id="$SESSION_ID" \
         --param namespace="$NAMESPACE" \
         --param agent_id="hypathia-$(printf '%03d' $i)" \
         --background &
done

# Wait for all to complete
wait
```

Or via Claude's native parallel execution:

```python
# In orchestrator context
agents = []
for i in range(TARGET_AGENTS):
    agent = spawn_claude_agent(
        model="sonnet-4.5",
        max_tokens=1_000_000,
        system_prompt=HYPATHIA_PROMPT,
        params={
            "session_id": session_id,
            "namespace": namespace,
            "agent_id": f"hypathia-{i+1:03d}"
        }
    )
    agents.append(agent)

# Orchestrator monitors until all complete
```

---

## MCP Embedding Server Architecture

### Server Status Response

```json
{
  "status": "loaded",
  "model_id": "Qwen/Qwen3-Embedding-8B",
  "device": "cpu",
  "max_seq_length": 32768,
  "embedding_dimension": 4096,
  "dtype": "torch.float32",
  "available_lenses": ["structural", "semantic", "behavioral"]
}
```

### Embedded Lens Instructions (CDATA Pattern)

The MCP server internally uses these instruction prompts as "gravitational lenses" that focus the embedding on specific aspects:

```xml
&lt;lenses&gt;
  &lt;lens name="structural"&gt;
    &lt;instruction&gt;&lt;![CDATA[
Embed the STRUCTURAL TOPOLOGY of code in a directed heterogeneous hypergraph.
Focus ONLY on:
- Graph connectivity (in-degree, out-degree, directed paths)
- Centrality measures (betweenness, pagerank, eigenvector)
- Community structure and clustering
- Node types (Controller, Service, Repository, Entity, Config)
- Edge types (METHOD_CALL, DEPENDENCY_INJECTION, DATA_FLOW)
- Hyperedge participation with source/target roles
- Design patterns and architectural motifs
Completely IGNORE what the code does - only HOW it's connected.
    ]]&gt;&lt;/instruction&gt;
  &lt;/lens&gt;
  
  &lt;lens name="semantic"&gt;
    &lt;instruction&gt;&lt;![CDATA[
Embed the SEMANTIC MEANING of Spring Boot code for CheckItOut.
Focus ONLY on:
- Business logic (influencer marketing, campaigns, payments)
- What this code DOES functionally
- Algorithms and data transformations
- Domain-specific terminology
- API contracts and interfaces
Completely IGNORE structure and runtime - only WHAT it means.
    ]]&gt;&lt;/instruction&gt;
  &lt;/lens&gt;
  
  &lt;lens name="behavioral"&gt;
    &lt;instruction&gt;&lt;![CDATA[
Embed the RUNTIME BEHAVIOR of code execution.
Focus ONLY on:
- State machines and transitions
- Error handling and recovery patterns
- Retry logic and circuit breakers
- Transaction boundaries
- Async operations and threading
- Side effects (DB writes, network calls, events)
- Causal relationships and downstream effects
Completely IGNORE static structure and meaning - only HOW it behaves.
    ]]&gt;&lt;/instruction&gt;
  &lt;/lens&gt;
&lt;/lenses&gt;
```

---

## XML Invocation Patterns

### Why Single Embedding Calls

Always generate ONE embedding at a time. Batch mode fills context too quickly with 4096-dimensional outputs (each embedding is ~32KB of float data when serialized).

### MCP Tool Call Examples

**Semantic Embedding (during file indexing):**

```python
# Pseudo-code showing the MCP call structure
mcp_call = {
    "tool": "qwen3-embedding:embed",
    "parameters": {
        "lens": "semantic",
        "dimension": 4096,
        "text": """
@Service
@Transactional
public class PaymentService {
    private final PaymentRepository paymentRepository;
    private final CampaignService campaignService;
    
    public PaymentResult processInfluencerPayment(Long campaignId, BigDecimal amount) {
        Campaign campaign = campaignService.findById(campaignId);
        Payment payment = Payment.builder()
            .campaign(campaign)
            .amount(amount)
            .status(PaymentStatus.PENDING)
            .build();
        return paymentRepository.save(payment);
    }
}
"""
    }
}
```

**Behavioral Embedding (during file indexing):**

```python
mcp_call = {
    "tool": "qwen3-embedding:embed",
    "parameters": {
        "lens": "behavioral",
        "dimension": 4096,
        "text": """
Runtime Analysis for PaymentService:
- Transaction boundary: Method-level @Transactional
- State transitions: PENDING -> PROCESSING -> COMPLETED/FAILED
- Side effects: Database write (paymentRepository.save)
- Dependencies called: campaignService.findById (read)
- Error paths: CampaignNotFoundException, PaymentProcessingException
- Retry logic: None
- Async: No
"""
    }
}
```

**Structural Embedding (during global synthesis):**

```python
mcp_call = {
    "tool": "qwen3-embedding:embed",
    "parameters": {
        "lens": "structural",
        "dimension": 4096,
        "text": """
Node: PaymentService
Node Type: SERVICE
Entity Type: STATE_MACHINE

Graph Properties:
- In-Degree: 3 (PaymentController, ScheduledPaymentJob, WebhookHandler)
- Out-Degree: 2 (PaymentRepository, CampaignService)
- Betweenness Centrality: 0.234
- PageRank: 0.0156
- Community: payment-cluster-7
- Erdős Number: 2

Hyperedge Participation:
- As Source: 2 hyperedges (ORCHESTRATION, DATA_FLOW)
- As Target: 3 hyperedges (REST_ENDPOINT, SCHEDULED_TASK, WEBHOOK)
- Types: [METHOD_CALL, DEPENDENCY_INJECTION, TRANSACTION_BOUNDARY]

Architectural Position:
- MVC Role: Service Layer
- Design Pattern: Transaction Script + Repository
- Layer: Business Logic (Level 2)
"""
    }
}
```

### Response Structure

```json
{
  "embedding": [0.0234, -0.0156, 0.0891, ...],  // 4096 floats
  "dimension": 4096,
  "lens": "semantic",
  "tokens_used": 156
}
```

---

## Stage 1: File Indexing with Semantic and Behavioral Embeddings

### Node Properties

Every file node MUST have:

| Property | Type | Description |
|----------|------|-------------|
| `file_path` | String | **Absolute path** on disk (e.g., `C:\Users\Norbert\...`) |
| `last_modified` | DateTime | File's last modification timestamp |
| `name` | String | File name without path |
| `content_hash` | String | SHA-256 of file content (for change detection) |
| `semantic_embedding` | Float[4096] | Semantic lens embedding |
| `behavioral_embedding` | Float[4096] | Behavioral lens embedding |
| `structural_embedding` | Float[4096] | Structural lens embedding (added in synthesis) |
| `node_type` | String | CONTROLLER, SERVICE, REPOSITORY, ENTITY, CONFIG, etc. |
| `entity_type` | String | STATE_MACHINE, CONFIG_STATE, SECURITY_LAYER, etc. |
| `indexed_at` | DateTime | When this node was indexed |
| `indexed_by` | String | Agent ID that indexed this file (for parallel tracking) |
| `namespace` | String | Graph namespace this file belongs to |

### Indexing Algorithm (Single File - Used by Hypathia)

```python
async def index_file(file_path: str, neo4j_client, agent_id: str, namespace: str) -> dict:
    """
    Index a single file with semantic and behavioral embeddings.
    Structural embedding is deferred to global synthesis.
    Called by individual Hypathia agents.
    """
    
    # 1. Read file metadata
    stat = os.stat(file_path)
    last_modified = datetime.fromtimestamp(stat.st_mtime)
    
    # 2. Read and hash content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    # 3. Detect node type from content/path
    node_type = detect_node_type(file_path, content)  # CONTROLLER, SERVICE, etc.
    entity_type = detect_entity_type(content)  # STATE_MACHINE, CONFIG_STATE, etc.
    
    # 4. Generate SEMANTIC embedding (what code does)
    semantic_emb = await mcp_embed(
        lens="semantic",
        text=content,
        dimension=4096
    )
    
    # 5. Extract behavioral context and generate BEHAVIORAL embedding
    behavioral_context = extract_behavioral_context(content)
    behavioral_emb = await mcp_embed(
        lens="behavioral",
        text=behavioral_context,
        dimension=4096
    )
    
    # 6. Create/Update node in Neo4j
    query = """
    MERGE (f:File {file_path: $file_path})
    SET f.name = $name,
        f.namespace = $namespace,
        f.last_modified = datetime($last_modified),
        f.content_hash = $content_hash,
        f.node_type = $node_type,
        f.entity_type = $entity_type,
        f.semantic_embedding = $semantic_embedding,
        f.behavioral_embedding = $behavioral_embedding,
        f.indexed_at = datetime(),
        f.indexed_by = $agent_id,
        f.needs_structural = true
    RETURN f
    """
    
    return await neo4j_client.query(query, {
        'file_path': file_path,
        'name': os.path.basename(file_path),
        'namespace': namespace,
        'last_modified': last_modified.isoformat(),
        'content_hash': content_hash,
        'node_type': node_type,
        'entity_type': entity_type,
        'semantic_embedding': semantic_emb,
        'behavioral_embedding': behavioral_emb,
        'agent_id': agent_id
    })


def extract_behavioral_context(content: str) -> str:
    """
    Extract runtime behavior patterns from code for behavioral embedding.
    """
    patterns = []
    
    # Transaction boundaries
    if '@Transactional' in content:
        patterns.append("Transaction boundary: @Transactional present")
    
    # State machines
    if 'enum' in content and 'Status' in content:
        patterns.append("State machine: Status enum detected")
    
    # Error handling
    try_count = content.count('try {') + content.count('try{')
    catch_count = content.count('catch (') + content.count('catch(')
    if try_count > 0:
        patterns.append(f"Error handling: {try_count} try blocks, {catch_count} catch blocks")
    
    # Async patterns
    if '@Async' in content or 'CompletableFuture' in content:
        patterns.append("Async: Asynchronous execution present")
    
    # Retry logic
    if '@Retry' in content or '@Retryable' in content:
        patterns.append("Retry: Retry logic configured")
    
    # Side effects
    if '.save(' in content or '.delete(' in content:
        patterns.append("Side effects: Database write operations")
    if 'RestTemplate' in content or 'WebClient' in content:
        patterns.append("Side effects: External HTTP calls")
    
    return f"""
Runtime Analysis for {extract_class_name(content)}:
{chr(10).join('- ' + p for p in patterns)}
"""
```

---

## Stage 2: Directed Heterogeneous Hypergraph Construction

### Hypergraph Model

```python
@dataclass
class DirectedHeterogeneousHypergraph:
    """
    H = (V, E, τ_v, τ_e, φ_src, φ_tgt)
    
    Where:
    - V: Set of nodes (files)
    - E: Set of hyperedges (multi-way relationships)
    - τ_v: V → T_v (node type function)
    - τ_e: E → T_e (edge type function)
    - φ_src: E → 2^V (source nodes function)
    - φ_tgt: E → 2^V (target nodes function)
    """
    
    node_types = {
        'CONTROLLER', 'SERVICE', 'REPOSITORY', 'ENTITY',
        'CONFIG', 'SECURITY', 'DTO', 'UTIL', 'TEST'
    }
    
    edge_types = {
        'METHOD_CALL', 'DEPENDENCY_INJECTION', 'INHERITANCE',
        'IMPLEMENTATION', 'REST_ENDPOINT', 'DATA_FLOW',
        'TRANSACTION_BOUNDARY', 'EVENT_EMISSION', 'EVENT_CONSUMPTION'
    }
```

### Neo4j Hyperedge Creation

```cypher
// Create directed hyperedge with source/target distinction
CREATE (he:Hyperedge {
  id: 'HE_' + randomUUID(),
  type: 'METHOD_CALL',
  edge_type: 'ORCHESTRATION',
  arity: 4,
  direction: 'DIRECTED',
  created_at: datetime()
})

// Connect SOURCE nodes (callers, triggers)
MATCH (controller:File {name: 'PaymentController'})
CREATE (controller)-[:IN_HYPEREDGE {
  role: 'CALLER', 
  direction: 'SOURCE', 
  weight: 1.0
}]->(he)

// Connect TARGET nodes (callees, effects)
MATCH (service:File {name: 'PaymentService'})
CREATE (service)-[:IN_HYPEREDGE {
  role: 'CALLEE', 
  direction: 'TARGET', 
  weight: 1.0
}]->(he)
```

---

## Stage 3: Global Synthesis with Structural Embeddings

### When to Run Global Synthesis

Global synthesis runs:
1. **After parallel indexing completes** (all Hypathia agents done)
2. After incremental reindexing completes
3. On-demand for specific subgraphs

**IMPORTANT**: Global Synthesis is SEQUENTIAL and runs AFTER all parallel indexing is complete.

### Synthesis Algorithm

```python
async def run_global_synthesis(neo4j_client, namespace: str):
    """
    Global synthesis computes graph metrics and generates structural embeddings.
    This captures the graph topology that individual files cannot see.
    
    MUST run after all parallel indexing is complete.
    """
    
    print("=== GLOBAL SYNTHESIS ===")
    
    # Step 1: Compute graph metrics using Neo4j GDS
    print("Step 1: Computing graph metrics...")
    await compute_graph_metrics(neo4j_client, namespace)
    
    # Step 2: Detect communities
    print("Step 2: Detecting communities...")
    await detect_communities(neo4j_client, namespace)
    
    # Step 3: Generate structural embeddings for all nodes
    print("Step 3: Generating structural embeddings...")
    await generate_structural_embeddings(neo4j_client, namespace)
    
    # Step 4: Mark synthesis complete
    await neo4j_client.query("""
        MATCH (t:IndexTracker {namespace: $namespace})
        SET t.last_synthesis = datetime(),
            t.synthesis_version = coalesce(t.synthesis_version, 0) + 1
    """, {'namespace': namespace})
    
    print("=== SYNTHESIS COMPLETE ===")


async def compute_graph_metrics(neo4j_client, namespace: str):
    """Compute centrality and other graph metrics."""
    
    # PageRank
    await neo4j_client.query("""
        CALL gds.pageRank.write({
            nodeProjection: 'File',
            relationshipProjection: 'CALLS',
            writeProperty: 'pagerank'
        })
    """)
    
    # Betweenness Centrality
    await neo4j_client.query("""
        CALL gds.betweenness.write({
            nodeProjection: 'File',
            relationshipProjection: 'CALLS',
            writeProperty: 'betweenness_centrality'
        })
    """)
    
    # Degree (in/out)
    await neo4j_client.query("""
        MATCH (f:File {namespace: $namespace})
        SET f.in_degree = size((f)<-[:CALLS]-()),
            f.out_degree = size((f)-[:CALLS]->())
    """, {'namespace': namespace})


async def generate_structural_embeddings(neo4j_client, namespace: str):
    """
    Generate structural embeddings for all nodes that need them.
    This uses graph context that's only available after full indexing.
    """
    
    # Get all nodes needing structural embedding
    nodes = await neo4j_client.query("""
        MATCH (f:File {namespace: $namespace})
        WHERE f.needs_structural = true
        RETURN f.file_path as file_path,
               f.name as name,
               f.node_type as node_type,
               f.entity_type as entity_type,
               f.in_degree as in_degree,
               f.out_degree as out_degree,
               f.pagerank as pagerank,
               f.betweenness_centrality as betweenness,
               f.community_id as community
    """, {'namespace': namespace})
    
    for node in nodes:
        # Build structural context from graph properties
        structural_context = build_structural_context(node, neo4j_client)
        
        # Generate structural embedding
        structural_emb = await mcp_embed(
            lens="structural",
            text=structural_context,
            dimension=4096
        )
        
        # Update node
        await neo4j_client.query("""
            MATCH (f:File {file_path: $file_path})
            SET f.structural_embedding = $embedding,
                f.needs_structural = false,
                f.structural_updated_at = datetime()
        """, {
            'file_path': node['file_path'],
            'embedding': structural_emb
        })


def build_structural_context(node: dict, neo4j_client) -> str:
    """
    Build the text context for structural embedding.
    This describes the node's position in the graph.
    """
    
    # Get hyperedge participation
    hyperedges = neo4j_client.query_sync("""
        MATCH (f:File {file_path: $path})-[r:IN_HYPEREDGE]->(he:Hyperedge)
        RETURN r.direction as direction, he.type as type
    """, {'path': node['file_path']})
    
    source_count = sum(1 for h in hyperedges if h['direction'] == 'SOURCE')
    target_count = sum(1 for h in hyperedges if h['direction'] == 'TARGET')
    hyperedge_types = list(set(h['type'] for h in hyperedges))
    
    return f"""
Node: {node['name']}
Node Type: {node['node_type']}
Entity Type: {node['entity_type']}

Graph Properties:
- In-Degree: {node['in_degree']}
- Out-Degree: {node['out_degree']}
- Betweenness Centrality: {node['betweenness']:.4f}
- PageRank: {node['pagerank']:.6f}
- Community: {node['community']}

Hyperedge Participation:
- As Source: {source_count} hyperedges
- As Target: {target_count} hyperedges
- Types: {hyperedge_types}

Architectural Position:
- MVC Role: {detect_mvc_role(node['node_type'])}
- Layer: {detect_layer(node['node_type'])}
"""
```

---

## Stage 4: Incremental Reindexing Algorithm

### Change Detection Strategy

```python
@dataclass
class FileChange:
    file_path: str
    change_type: str  # 'ADDED', 'MODIFIED', 'DELETED'
    old_hash: str = None
    new_hash: str = None


async def detect_changes(project_path: str, neo4j_client, namespace: str) -> list[FileChange]:
    """
    Compare filesystem state with Neo4j index to detect changes.
    """
    changes = []
    
    # Get all indexed files from Neo4j
    indexed = await neo4j_client.query("""
        MATCH (f:File {namespace: $namespace})
        RETURN f.file_path as path, f.content_hash as hash
    """, {'namespace': namespace})
    indexed_map = {r['path']: r['hash'] for r in indexed}
    
    # Scan filesystem
    current_files = set()
    for root, dirs, files in os.walk(project_path):
        # Skip common non-code directories
        dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', 'target', 'build'}]
        
        for file in files:
            if not file.endswith(('.java', '.xml', '.yml', '.yaml', '.properties')):
                continue
                
            file_path = os.path.join(root, file)
            current_files.add(file_path)
            
            # Compute current hash
            with open(file_path, 'rb') as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            
            if file_path not in indexed_map:
                # NEW file
                changes.append(FileChange(
                    file_path=file_path,
                    change_type='ADDED',
                    new_hash=current_hash
                ))
            elif indexed_map[file_path] != current_hash:
                # MODIFIED file
                changes.append(FileChange(
                    file_path=file_path,
                    change_type='MODIFIED',
                    old_hash=indexed_map[file_path],
                    new_hash=current_hash
                ))
    
    # Check for DELETED files
    for indexed_path in indexed_map:
        if indexed_path not in current_files:
            changes.append(FileChange(
                file_path=indexed_path,
                change_type='DELETED',
                old_hash=indexed_map[indexed_path]
            ))
    
    return changes
```

### Handling Each Change Type

```python
async def process_changes(changes: list[FileChange], neo4j_client, namespace: str):
    """
    Process detected changes and trigger appropriate reindexing.
    For large change sets, use parallel Hypathia agents.
    """
    
    added = [c for c in changes if c.change_type == 'ADDED']
    modified = [c for c in changes if c.change_type == 'MODIFIED']
    deleted = [c for c in changes if c.change_type == 'DELETED']
    
    print(f"Changes detected: {len(added)} added, {len(modified)} modified, {len(deleted)} deleted")
    
    # 1. Handle DELETED files (fast, do synchronously)
    for change in deleted:
        await handle_deleted_file(change, neo4j_client)
    
    # 2. For ADDED and MODIFIED: use parallel indexing if > 10 files
    to_index = added + modified
    if len(to_index) > 10:
        print(f"Large change set ({len(to_index)} files), using parallel Hypathia agents...")
        await parallel_index_changes(to_index, neo4j_client, namespace)
    else:
        # Small change set, process sequentially
        for change in to_index:
            await index_file(change.file_path, neo4j_client, 'master', namespace)
    
    # 3. Run global synthesis if any changes occurred
    if changes:
        print("Running global synthesis to update structural embeddings...")
        await run_global_synthesis(neo4j_client, namespace)


async def handle_deleted_file(change: FileChange, neo4j_client):
    """
    Handle a deleted file:
    1. Remove the node
    2. Remove all relationships
    3. Clean up orphaned hyperedges
    """
    
    print(f"  DELETED: {change.file_path}")
    
    # Remove node and all its relationships
    await neo4j_client.query("""
        MATCH (f:File {file_path: $path})
        
        // Remove hyperedge participation
        OPTIONAL MATCH (f)-[r:IN_HYPEREDGE]->(he:Hyperedge)
        DELETE r
        
        // Clean up orphaned hyperedges (no remaining participants)
        WITH f, collect(he) as hyperedges
        UNWIND hyperedges as he
        OPTIONAL MATCH (other:File)-[:IN_HYPEREDGE]->(he)
        WITH f, he, count(other) as remaining
        WHERE remaining = 0
        DELETE he
        
        // Remove all other relationships and the node
        WITH f
        DETACH DELETE f
    """, {'path': change.file_path})
```

### Weekly Reindexing Orchestration

```python
async def weekly_reindex(project_path: str, namespace: str, neo4j_client):
    """
    Full weekly reindexing workflow:
    1. Detect all changes since last index
    2. Process changes (parallel if large)
    3. Run global synthesis
    4. Validate graph integrity
    """
    
    print("=" * 60)
    print(f"WEEKLY REINDEX: {namespace}")
    print(f"Project: {project_path}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Step 1: Detect changes
    print("\n[1/4] Detecting changes...")
    changes = await detect_changes(project_path, neo4j_client, namespace)
    
    if not changes:
        print("No changes detected. Skipping reindex.")
        return
    
    # Step 2: Process changes (uses parallel Hypathia if needed)
    print(f"\n[2/4] Processing {len(changes)} changes...")
    await process_changes(changes, neo4j_client, namespace)
    
    # Step 3: Global synthesis runs automatically in process_changes
    
    # Step 4: Validate
    print("\n[4/4] Validating graph integrity...")
    await validate_graph_integrity(neo4j_client, namespace)
    
    print("\n" + "=" * 60)
    print("WEEKLY REINDEX COMPLETE")
    print("=" * 60)


async def validate_graph_integrity(neo4j_client, namespace: str):
    """
    Validate the graph is in a consistent state.
    """
    
    # Check for nodes missing embeddings
    missing = await neo4j_client.query("""
        MATCH (f:File {namespace: $namespace})
        WHERE f.semantic_embedding IS NULL 
           OR f.behavioral_embedding IS NULL
           OR f.structural_embedding IS NULL
        RETURN count(f) as count
    """, {'namespace': namespace})
    
    if missing[0]['count'] > 0:
        print(f"  WARNING: {missing[0]['count']} nodes missing embeddings")
    
    # Check for orphaned hyperedges
    orphans = await neo4j_client.query("""
        MATCH (he:Hyperedge)
        WHERE NOT (he)<-[:IN_HYPEREDGE]-()
        RETURN count(he) as count
    """)
    
    if orphans[0]['count'] > 0:
        print(f"  WARNING: {orphans[0]['count']} orphaned hyperedges")
        # Clean them up
        await neo4j_client.query("""
            MATCH (he:Hyperedge)
            WHERE NOT (he)<-[:IN_HYPEREDGE]-()
            DELETE he
        """)
    
    print("  Validation complete.")
```

---

## Stage 5: The Erdős Agent Navigation

### Triple-Lens Query

```python
class ErdosAgent:
    """
    Navigate the codebase using all three embedding lenses.
    """
    
    def __init__(self, neo4j_client, mcp_client):
        self.neo4j = neo4j_client
        self.mcp = mcp_client
    
    async def find_relevant_files(
        self,
        query: str,
        namespace: str,
        weights: dict = None
    ) -> list[dict]:
        """
        Find files relevant to a query using weighted triple similarity.
        
        Default weights favor semantic (what code does) but include
        structural (how connected) and behavioral (how runs).
        """
        
        weights = weights or {
            'semantic': 0.50,
            'structural': 0.25,
            'behavioral': 0.25
        }
        
        # Generate query embeddings for each lens
        sem_query = await self.mcp.embed(lens='semantic', text=query)
        struct_query = await self.mcp.embed(lens='structural', text=query)
        behav_query = await self.mcp.embed(lens='behavioral', text=query)
        
        # Weighted similarity search
        results = await self.neo4j.query("""
            MATCH (f:File {namespace: $namespace})
            WHERE f.semantic_embedding IS NOT NULL
            
            WITH f,
                 $w_sem * gds.similarity.cosine(f.semantic_embedding, $sem_q) +
                 $w_struct * gds.similarity.cosine(f.structural_embedding, $struct_q) +
                 $w_behav * gds.similarity.cosine(f.behavioral_embedding, $behav_q) 
                 AS score
            
            WHERE score > 0.5
            
            RETURN f.file_path as path,
                   f.name as name,
                   f.node_type as type,
                   score
            ORDER BY score DESC
            LIMIT 20
        """, {
            'namespace': namespace,
            'sem_q': sem_query,
            'struct_q': struct_query,
            'behav_q': behav_query,
            'w_sem': weights['semantic'],
            'w_struct': weights['structural'],
            'w_behav': weights['behavioral']
        })
        
        return results
```

---

## Neo4j Schema and Queries

### Complete Schema

```cypher
// Node constraints
CREATE CONSTRAINT file_path_unique IF NOT EXISTS
FOR (f:File) REQUIRE f.file_path IS UNIQUE;

CREATE CONSTRAINT hyperedge_id_unique IF NOT EXISTS
FOR (he:Hyperedge) REQUIRE he.id IS UNIQUE;

// IndexTracker constraints (for parallel indexing)
CREATE CONSTRAINT indextracker_session_unique IF NOT EXISTS
FOR (t:IndexTracker) REQUIRE t.session_id IS UNIQUE;

CREATE CONSTRAINT filetask_path_unique IF NOT EXISTS
FOR (ft:FileTask) REQUIRE ft.file_path IS UNIQUE;

// Indexes for common queries
CREATE INDEX file_hash IF NOT EXISTS FOR (f:File) ON (f.content_hash);
CREATE INDEX file_type IF NOT EXISTS FOR (f:File) ON (f.node_type);
CREATE INDEX file_namespace IF NOT EXISTS FOR (f:File) ON (f.namespace);
CREATE INDEX file_needs_structural IF NOT EXISTS FOR (f:File) ON (f.needs_structural);
CREATE INDEX hyperedge_type IF NOT EXISTS FOR (he:Hyperedge) ON (he.type);

// IndexTracker indexes
CREATE INDEX filetask_status IF NOT EXISTS FOR (ft:FileTask) ON (ft.status);
CREATE INDEX filetask_agent IF NOT EXISTS FOR (ft:FileTask) ON (ft.claimed_by);
CREATE INDEX filetask_session IF NOT EXISTS FOR (ft:FileTask) ON (ft.session_id);

// Vector indexes for similarity search
CALL db.index.vector.createNodeIndex(
  'semantic_index',
  'File',
  'semantic_embedding',
  4096,
  'cosine'
);

CALL db.index.vector.createNodeIndex(
  'structural_index',
  'File',
  'structural_embedding',
  4096,
  'cosine'
);

CALL db.index.vector.createNodeIndex(
  'behavioral_index',
  'File',
  'behavioral_embedding',
  4096,
  'cosine'
);
```

### Sample Queries

**Find files by semantic similarity:**

```cypher
CALL db.index.vector.queryNodes(
  'semantic_index',
  10,
  $query_embedding
) YIELD node, score
RETURN node.file_path, node.name, score
```

**Get all files modified after a date:**

```cypher
MATCH (f:File {namespace: $namespace})
WHERE f.last_modified > datetime($since)
RETURN f.file_path, f.last_modified
ORDER BY f.last_modified DESC
```

**Find structural neighbors of a file:**

```cypher
MATCH (f:File {file_path: $path})-[:CALLS*1..2]-(neighbor:File)
RETURN DISTINCT neighbor.file_path, neighbor.node_type
```

**Monitor parallel indexing progress:**

```cypher
MATCH (t:IndexTracker {session_id: $session_id})
MATCH (ft:FileTask {session_id: $session_id})
WITH t, 
     sum(CASE WHEN ft.status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
     sum(CASE WHEN ft.status = 'FAILED' THEN 1 ELSE 0 END) as failed,
     sum(CASE WHEN ft.status = 'PROCESSING' THEN 1 ELSE 0 END) as processing,
     sum(CASE WHEN ft.status = 'PENDING' THEN 1 ELSE 0 END) as pending
RETURN t.total_files as total, completed, failed, processing, pending,
       round(100.0 * (completed + failed) / t.total_files, 2) as progress_percent
```

---

## Implementation Code

### Main Pipeline Entry Point

```python
import asyncio
from datetime import datetime

async def main():
    """
    Main entry point for the triple-lens pipeline.
    """
    
    # Configuration
    config = {
        'root_paths': [
            r'C:\Users\Norbert\IdeaProjects\CheckItOut',
            r'C:\Users\Norbert\IdeaProjects\CheckItOut-Frontend'
        ],
        'namespace': 'checkitout',
        'neo4j_uri': 'bolt://localhost:7687',
        'neo4j_user': 'neo4j',
        'neo4j_password': 'password',
        'target_agents': 4,  # Number of parallel Hypathia agents
        'file_extensions': ['.java', '.xml', '.yml', '.yaml', '.properties'],
        'exclude_dirs': ['.git', 'node_modules', 'target', 'build', '.idea']
    }
    
    # Initialize clients
    neo4j_client = Neo4jClient(
        config['neo4j_uri'],
        config['neo4j_user'],
        config['neo4j_password']
    )
    
    mcp_client = MCPEmbeddingClient()
    
    # Run pipeline
    print("Triple-Lens Pipeline v7.0 - Parallel Hypathia")
    print("=" * 50)
    
    # Option 1: Full parallel index
    await full_parallel_index(config, neo4j_client)
    
    # Option 2: Incremental weekly reindex
    # await weekly_reindex(
    #     config['root_paths'][0],
    #     config['namespace'],
    #     neo4j_client
    # )


if __name__ == '__main__':
    asyncio.run(main())
```

### MCP Client Wrapper

```python
class MCPEmbeddingClient:
    """
    Wrapper for MCP Qwen3-Embedding calls.
    Always use single embeddings (not batch) to avoid context overflow.
    """
    
    async def embed(self, lens: str, text: str, dimension: int = 4096) -> list[float]:
        """
        Generate a single embedding using the specified lens.
        
        Args:
            lens: One of 'structural', 'semantic', 'behavioral'
            text: The text to embed
            dimension: Embedding dimension (default 4096)
        
        Returns:
            List of floats (the embedding vector)
        """
        
        # This would be the actual MCP call
        # In practice, this is invoked via Claude's function calling
        result = await mcp_call(
            tool="qwen3-embedding:embed",
            parameters={
                "lens": lens,
                "text": text,
                "dimension": dimension
            }
        )
        
        return result['embedding']
    
    async def model_info(self) -> dict:
        """Get model status and available lenses."""
        return await mcp_call(tool="qwen3-embedding:model_info")
```

---

## Summary

### Pipeline Flow (v7.0 with Parallel Hypathia)

```
┌─────────────────────────────────────────────────────────────────┐
│              MASTER ORCHESTRATOR (claude.md)                     │
├─────────────────────────────────────────────────────────────────┤
│  1. Receive ROOT_PATHS, NAMESPACE, TARGET_AGENTS                │
│  2. Scan directories, discover files                            │
│  3. Populate IndexTracker (FileTask for each file)              │
│  4. Spawn N Hypathia agents                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                PARALLEL INDEXING (Hypathia Agents)               │
├─────────────────────────────────────────────────────────────────┤
│  Each agent (loop):                                              │
│    1. Claim next PENDING file (atomic)                          │
│    2. Read file via MCP Filesystem                              │
│    3. Generate SEMANTIC embedding                               │
│    4. Generate BEHAVIORAL embedding                             │
│    5. Write node to Neo4j                                       │
│    6. Mark FileTask COMPLETED                                   │
│    7. Repeat until no PENDING files                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            GLOBAL SYNTHESIS (Sequential, after parallel)         │
├─────────────────────────────────────────────────────────────────┤
│  1. Compute graph metrics (PageRank, Betweenness, etc.)         │
│  2. Detect communities                                           │
│  3. For each node:                                               │
│     - Build structural context from graph properties             │
│     - Generate STRUCTURAL embedding                              │
│  4. Mark synthesis complete                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                RESUME CAPABILITY                                 │
├─────────────────────────────────────────────────────────────────┤
│  On restart:                                                     │
│  • COMPLETED files: Skip (already indexed)                       │
│  • PENDING files: Available for agents                           │
│  • CLAIMED/PROCESSING files: Reset to PENDING after timeout      │
│  • FAILED files: Retry if retry_count < 3                        │
└─────────────────────────────────────────────────────────────────┘
```

### Key Properties on Every Node

| Property | Source | When Set |
|----------|--------|----------|
| `file_path` | Filesystem | Indexing |
| `last_modified` | Filesystem | Indexing |
| `content_hash` | SHA-256 | Indexing |
| `namespace` | Config | Indexing |
| `semantic_embedding` | MCP:semantic | Parallel Indexing |
| `behavioral_embedding` | MCP:behavioral | Parallel Indexing |
| `structural_embedding` | MCP:structural | Global Synthesis |
| `node_type` | Code analysis | Indexing |
| `indexed_by` | Agent ID | Parallel Indexing |
| `pagerank` | Neo4j GDS | Global Synthesis |
| `community_id` | Neo4j GDS | Global Synthesis |

### IndexTracker Status Flow

```
FileTask Status:
  PENDING ──────────────────────────────────────────┐
     │                                              │
     │ (agent claims)                               │ (timeout recovery)
     ▼                                              │
  CLAIMED ──────────────────────────────────────────┤
     │                                              │
     │ (start processing)                           │
     ▼                                              │
  PROCESSING ───────────────────────────────────────┘
     │
     ├─── (success) ───▶ COMPLETED
     │
     └─── (error) ─────▶ FAILED
                            │
                            │ (retry if count < 3)
                            ▼
                         PENDING
```

---

**Version 7.0.0 - Parallel Hypathia Edition**
