# Information Lensing Reranker

Domain-tuned MCP server for semantic similarity scoring in enterprise codebases.

## Simple Interface

```
Input:  (code_a, code_b)
Output: probability score (0.0 - 1.0)
```

**Score interpretation:**
- **0.7-1.0**: Same semantic domain (both handle payments, both handle users)
- **0.3-0.7**: Related domains (PaymentService ↔ PaymentController)
- **0.0-0.3**: Different domains despite structural similarity ← **Key for Information Lensing**

## What It Does

Generic embeddings suffer from **semantic collapse** - structurally similar code gets similar embeddings even when semantically different:

```
PaymentService.java  ↔  InventoryService.java
Embedding similarity:    0.94 (HIGH - sees Spring patterns)
Reranker score:          0.08 (LOW - different business domains)
Divergence:              0.86 → Lens needs to correct this!
```

This server uses a custom instruction that transforms Qwen3-Reranker from "relevance detector" to "semantic domain similarity detector".

## Supported Tech Stacks

- **Java Spring**: Services, Controllers, Repositories, Entities, DTOs, Configs
- **Angular**: Components, Services, Modules, Guards, Interceptors, NgRx
- **Ansible**: Playbooks, Roles, Tasks, Templates
- **CI/CD**: GitHub Actions, GitLab CI, Jenkins pipelines
- **Configs**: YAML, JSON, properties, env files

## Usage

### With Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "information-lensing-reranker": {
      "command": "uv",
      "args": ["--directory", "C:/path/to/McpServerForReranking", "run", "qwen3-reranker-mcp"]
    }
  }
}
```

### Programmatic (Python)

```python
from qwen3_reranker_mcp.reranker_engine import get_engine

engine = get_engine()

# Score a pair
result = engine.score_pair(
    query=open("PaymentService.java").read(),
    document=open("InventoryService.java").read(),
)

print(f"Score: {result.score}")  # 0.08 - different domains

# Or just get the score directly
score = engine.score_pair_raw(code_a, code_b)
```

## Information Lensing Workflow

```python
# 1. Get embeddings (your embedding server)
embeddings = embedding_model.embed_files(all_files)

# 2. Find high-similarity pairs from embeddings
candidate_pairs = find_pairs_above_threshold(embeddings, threshold=0.85)

# 3. Score each pair with reranker
for i, j in candidate_pairs:
    emb_sim = cosine_similarity(embeddings[i], embeddings[j])
    reranker_score = engine.score_pair_raw(files[i], files[j])
    
    divergence = emb_sim - reranker_score
    
    if divergence > 0.4:
        # This pair needs lens correction!
        # Store in Neo4j, add to training data, etc.
        print(f"High divergence: {files[i]} ↔ {files[j]}")
        print(f"  Embedding: {emb_sim:.2f}, Reranker: {reranker_score:.2f}")
```

## Configuration

Environment variables (prefix: `QWEN3_RERANKER_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-Reranker-8B` | HuggingFace model ID |
| `DEVICE` | `cpu` | Device (cpu/cuda/mps/auto) |
| `TORCH_DTYPE` | `bfloat16` | Precision (float32/float16/bfloat16) |
| `MAX_LENGTH` | `8192` | Max sequence length |
| `USE_CUSTOM_INSTRUCTION` | `true` | Use domain-tuned instruction |

## Custom Instruction

The key innovation is the system instruction that shifts scoring from "relevance" to "semantic domain similarity":

```
FOCUS ON:
1. Business entity/process being handled (Payment? User? Order? Inventory?)
2. Would a product requirement change affect BOTH segments?
3. Data flow dependencies between segments

IGNORE:
- @Service/@Repository/@Controller annotations
- Similar Spring patterns (autowiring, transaction management)
- Generic naming conventions (ServiceImpl, Repository, etc.)
```

You can customize `QWEN3_RERANKER_CUSTOM_INSTRUCTION` for your specific codebase.

## Requirements

- Python 3.11+
- ~32GB RAM for CPU inference (bfloat16)
- ~20GB VRAM for GPU inference

## Installation

```bash
cd McpServerForReranking
uv sync
```

## License

MIT
