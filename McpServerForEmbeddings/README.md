# Qwen3-Embedding MCP Server with Information Lensing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)

A high-performance [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server providing state-of-the-art text embeddings using **Qwen3-Embedding-8B** with **Information Lensing** — domain-specific embedding transformation for enterprise codebases.

## What is Information Lensing?

Just as gravitational lensing bends light to reveal distant galaxies, **Information Lensing** curves embedding space to reveal hidden semantic structure in code.

Generic embeddings suffer from **semantic collapse** — `PaymentService` and `InventoryService` look nearly identical (0.94 cosine similarity) despite having completely different business purposes.

Information Lensing applies three "gravitational lenses" that focus embeddings on different aspects:

| Lens | Focus | Use Case |
|------|-------|----------|
| **structural** | Graph topology, architectural position, connectivity | Finding related modules, dependency analysis |
| **semantic** | Business logic, domain concepts, code meaning | Finding code by functionality |
| **behavioral** | Runtime patterns, state machines, side effects | Finding code by execution behavior |

Reference: Marchewka (2025), [Information Lensing: A Metric Learning Approach to Domain-Specific Embedding Transformation](../GraphTheoryInSystemModeling/Appendix_C_Information_Lensing.md).

> **Status (2026-10).** The server registers three tools: `embed`, `batch_embed` and `model_info`. The
> `embed_triple` and `similarity` tools described below belong to an earlier design and are not in the
> current server: call `embed` once per lens, and use the [reranking server](../McpServerForReranking/) to
> compare a query with documents.

## Features

- 🔭 **Information Lensing**: Three domain-specific lenses (structural, semantic, behavioral)
- 🚀 **State-of-the-Art Quality**: Qwen3-Embedding-8B ranks #1 on MTEB multilingual benchmark
- 🌍 **100+ Languages**: Including programming languages (Java, TypeScript, Python, etc.)
- 📏 **Flexible Dimensions**: MRL for custom dimensions (128-4096)
- 📄 **32K Context**: Process long files up to 32,768 tokens
- 💻 **CPU Optimized**: Designed for high-RAM CPU systems (64GB+ recommended)

## Requirements

- **Python**: 3.10 or higher
- **RAM**: 32GB minimum, 64GB+ recommended
- **Disk**: ~16GB for model weights

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/Check-It-Out-Dev/graph-theory-system-modeling.git
cd graph-theory-system-modeling/McpServerForEmbeddings
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e .
```

### Running

```bash
python -m qwen3_embedding_mcp
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "qwen3-embedding": {
      "command": "C:/path/to/.venv/Scripts/python.exe",
      "args": ["-m", "qwen3_embedding_mcp"]
    }
  }
}
```

## Available Tools

### `embed` — Single Document with Lens

Generate embedding for a document using Information Lensing.

```json
{
  "text": "class PaymentService { ... }",
  "lens": "semantic"
}
```

**Parameters:**
- `text` (required): Document content (max 32K tokens)
- `lens` (required): `"structural"`, `"semantic"`, or `"behavioral"`
- `dimension` (optional): Output dimension 128-4096 (default: 4096)

**Returns:** 4096-dimensional embedding focused on the selected aspect.

---

### `embed_triple` — All Three Lenses

Generate all three lens embeddings in one call. Use for building the triple-embedded hypergraph.

```json
{
  "text": "class PaymentService { ... }"
}
```

**Returns:**
```json
{
  "structural": { "embedding": [...], "dimensions": 4096 },
  "semantic": { "embedding": [...], "dimensions": 4096 },
  "behavioral": { "embedding": [...], "dimensions": 4096 }
}
```

---

### `batch_embed` — Multiple Documents, Same Lens

```json
{
  "texts": ["class A { ... }", "class B { ... }"],
  "lens": "semantic"
}
```

---

### `similarity` — Compare Queries to Documents

```json
{
  "queries": ["payment processing"],
  "documents": ["PaymentService.java content...", "InventoryService.java content..."],
  "query_lens": "semantic"
}
```

---

### `model_info` — Status and Available Lenses

```json
{}
```

## Domain Instructions (Information Lensing)

The three lenses apply these domain-specific instructions automatically:

### Structural Lens
> Embed the STRUCTURAL TOPOLOGY of code in a directed heterogeneous hypergraph. Focus ONLY on: graph connectivity, centrality measures, community structure, node types (Controller, Service, Repository), edge types (METHOD_CALL, DEPENDENCY_INJECTION), design patterns. Completely IGNORE what the code does - only HOW it's connected.

### Semantic Lens
> Embed the SEMANTIC MEANING of Spring Boot code for CheckItOut. Focus ONLY on: business logic (influencer marketing, campaigns, payments), what this code DOES functionally, algorithms, domain terminology, API contracts. Completely IGNORE structure and runtime - only WHAT it means.

### Behavioral Lens
> Embed the RUNTIME BEHAVIOR of code execution. Focus ONLY on: state machines, error handling, retry logic, circuit breakers, transaction boundaries, async operations, side effects (DB writes, network calls), causal relationships. Completely IGNORE static structure and meaning - only HOW it behaves.

## Usage Example

```
User: Embed this PaymentService code with semantic lens

Claude: [calls embed with lens="semantic"]
Result: 4096D embedding focused on business meaning
```

```
User: Generate all three embeddings for InventoryController

Claude: [calls embed_triple]
Result: structural, semantic, and behavioral embeddings
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_EMBEDDING_MODEL_ID` | `Qwen/Qwen3-Embedding-8B` | Model ID |
| `QWEN3_EMBEDDING_DEVICE` | `cpu` | Device: cpu, cuda, mps |
| `QWEN3_EMBEDDING_DEFAULT_DIMENSION` | `4096` | Default embedding dimension |
| `QWEN3_EMBEDDING_LOG_LEVEL` | `INFO` | Logging level |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Client (Claude)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (stdio)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Tool Router                        │   │
│  │  embed | embed_triple | batch_embed | similarity    │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                    │
│  ┌─────────────────────▼───────────────────────────────┐   │
│  │              Embedding Engine                        │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │         Information Lensing Layer           │    │   │
│  │  │  structural | semantic | behavioral         │    │   │
│  │  └─────────────────────┬───────────────────────┘    │   │
│  │                        │                             │   │
│  │  ┌─────────────────────▼───────────────────────┐    │   │
│  │  │          Qwen3-Embedding-8B                 │    │   │
│  │  │          (sentence-transformers)            │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Performance

- **Single embed**: ~200-500ms (CPU)
- **Triple embed**: ~600-1500ms (CPU)
- **Batch of 20**: ~2-4s (CPU)
- **Cold start**: ~2-3 minutes (model loading)

## Related Work

- [Information Lensing Appendix](../GraphTheoryInSystemModeling/Appendix_C_Information_Lensing.md)
- [Triple-lens pipeline (working note, superseded)](../WorkingNotes/triple_lens_graph_pipeline.md)
- [Qwen3-Embedding Paper](https://arxiv.org/abs/2506.05176)

---

Made with ❤️ by [Norbert Marchewka](https://www.linkedin.com/in/norbert-marchewka-292377129/)
