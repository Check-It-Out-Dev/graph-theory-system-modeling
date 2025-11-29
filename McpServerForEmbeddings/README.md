# Qwen3-Embedding MCP Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)

A high-performance [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server providing state-of-the-art text embeddings using **Qwen3-Embedding-8B** — the #1 ranked model on the MTEB multilingual leaderboard (score 70.58, as of June 2025).

## Features

- 🚀 **State-of-the-Art Quality**: Qwen3-Embedding-8B ranks #1 on MTEB multilingual benchmark
- 🌍 **100+ Languages**: Comprehensive multilingual and cross-lingual support
- 📏 **Flexible Dimensions**: Matryoshka Representation Learning (MRL) for custom dimensions (128-4096)
- 🎯 **Instruction-Aware**: Built-in `query`/`passage` prompts + custom instructions (1-5% boost)
- 📄 **32K Context**: Process long documents up to 32,768 tokens
- 💻 **CPU Optimized**: Designed for high-RAM CPU systems (64GB+ recommended)
- 🔌 **MCP Native**: Seamless integration with Claude Desktop, Cursor, and other MCP clients

## Requirements

- **Python**: 3.10 or higher
- **RAM**: 32GB minimum, 64GB+ recommended for optimal performance
- **Disk**: ~16GB for model weights (cached after first download)
- **OS**: Windows, macOS, or Linux

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/NorbertMarchewka/qwen3-embedding-mcp.git
cd qwen3-embedding-mcp

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Running the Server

```bash
# Run directly
python -m qwen3_embedding_mcp

# Or use the entry point
qwen3-embedding-mcp
```

The server will:
1. Load the Qwen3-Embedding-8B model (~1-3 minutes on first run)
2. Cache model weights for faster subsequent starts
3. Listen for MCP connections via stdio

### Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "qwen3-embedding": {
      "command": "python",
      "args": ["-m", "qwen3_embedding_mcp"],
      "cwd": "C:/path/to/qwen3-embedding-mcp",
      "env": {
        "QWEN3_EMBEDDING_DEVICE": "cpu",
        "QWEN3_EMBEDDING_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**Windows with virtual environment:**

```json
{
  "mcpServers": {
    "qwen3-embedding": {
      "command": "C:/path/to/qwen3-embedding-mcp/.venv/Scripts/python.exe",
      "args": ["-m", "qwen3_embedding_mcp"],
      "env": {
        "QWEN3_EMBEDDING_DEVICE": "cpu"
      }
    }
  }
}
```

## Available Tools

### `embed`

Generate embeddings for one or more texts.

```json
{
  "texts": "What is machine learning?",
  "is_query": true
}
```

Or for documents:
```json
{
  "texts": ["Machine learning is...", "Deep learning uses..."],
  "prompt_name": "passage"
}
```

With custom instruction:
```json
{
  "texts": "quantum entanglement applications",
  "instruction": "Given a scientific query, retrieve relevant research papers",
  "dimension": 1024
}
```

**Parameters:**
- `texts` (required): String or array of strings to embed (max 32K tokens each)
- `is_query` (optional): Set true for search queries (uses built-in query prompt)
- `prompt_name` (optional): Built-in prompt - "query" or "passage" (recommended)
- `instruction` (optional): Custom instruction (overrides prompt_name)
- `dimension` (optional): Output dimension, 128-4096 (default: 4096)
- `normalize` (optional): L2-normalize embeddings (default: true)

### `similarity`

Compute semantic similarity between queries and documents.

```json
{
  "queries": ["What is the capital of France?"],
  "documents": [
    "Paris is the capital of France.",
    "London is the capital of England.",
    "Berlin is the capital of Germany."
  ],
  "query_instruction": "Given a question, retrieve the answer"
}
```

**Returns:** Similarity matrix and ranked results.

### `model_info`

Get information about the loaded model.

```json
{}
```

**Returns:** Model status, device, dimensions, and configuration.

## Configuration

All settings can be configured via environment variables with the `QWEN3_EMBEDDING_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_EMBEDDING_MODEL_ID` | `Qwen/Qwen3-Embedding-8B` | HuggingFace model ID |
| `QWEN3_EMBEDDING_DEVICE` | `cpu` | Device: `cpu`, `cuda`, `mps`, `auto` |
| `QWEN3_EMBEDDING_TORCH_DTYPE` | `float32` | Model precision |
| `QWEN3_EMBEDDING_DEFAULT_DIMENSION` | `4096` | Default embedding dimension |
| `QWEN3_EMBEDDING_MAX_SEQUENCE_LENGTH` | `8192` | Max input tokens |
| `QWEN3_EMBEDDING_BATCH_SIZE` | `8` | Batch size for processing |
| `QWEN3_EMBEDDING_LOG_LEVEL` | `INFO` | Logging verbosity |
| `QWEN3_EMBEDDING_CACHE_DIR` | (HF default) | Custom cache directory |

You can also use a `.env` file in the project root.

## Usage Examples

### Semantic Search

```
User: Use the embed tool to create embeddings for my search query "best restaurants in Warsaw"

Claude: [calls embed tool with instruction for retrieval]
```

### Document Similarity

```
User: Compare these two documents for similarity using the similarity tool

Claude: [calls similarity tool, returns similarity score]
```

### RAG Pipeline Integration

The embeddings are perfect for building RAG pipelines:

1. **Indexing**: Embed documents without instruction
2. **Querying**: Embed queries with retrieval instruction
3. **Retrieval**: Use similarity or vector database
4. **Generation**: Pass retrieved context to LLM

## Performance

### Memory Usage
- Model weights: ~32GB RAM (float32)
- Runtime overhead: ~2-4GB
- Total recommended: 64GB+

### Latency (CPU, Ryzen 9 9950X)
- Single text: ~200-500ms
- Batch of 8: ~1-2s
- First request (cold start): ~2-3 minutes

### Embedding Quality
- MTEB Multilingual: 70.58 (#1 as of June 2025)
- MTEB English: Competitive with top models
- Code Retrieval: Strong performance

## Troubleshooting

### Model Loading Slow

First load downloads ~16GB. Subsequent loads use cache:
```bash
# Custom cache directory
export QWEN3_EMBEDDING_CACHE_DIR=/path/to/cache
```

### Out of Memory

For systems with <64GB RAM:
```bash
# Use smaller model
export QWEN3_EMBEDDING_MODEL_ID=Qwen/Qwen3-Embedding-0.6B

# Or use smaller dimensions
export QWEN3_EMBEDDING_DEFAULT_DIMENSION=1024
```

### CUDA Out of Memory

The 8B model requires ~16GB VRAM. For smaller GPUs:
```bash
export QWEN3_EMBEDDING_DEVICE=cpu
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM) for the Qwen3-Embedding models
- [Anthropic](https://anthropic.com) for the Model Context Protocol
- [Sentence Transformers](https://sbert.net) for the excellent library

## Related Projects

- [Qwen3-Reranker MCP](https://github.com/NorbertMarchewka/qwen3-reranker-mcp) - Companion reranking server
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) - Official MCP SDK
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) - Embedding library

---

Made with ❤️ by [Norbert Marchewka](https://github.com/NorbertMarchewka)
