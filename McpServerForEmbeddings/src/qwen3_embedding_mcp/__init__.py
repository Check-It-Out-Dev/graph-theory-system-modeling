"""
Qwen3-Embedding-8B MCP Server

A high-performance Model Context Protocol server providing state-of-the-art
text embeddings using Qwen3-Embedding-8B model.

Features:
- Single text and batch embedding generation
- Configurable embedding dimensions (Matryoshka Representation Learning)
- Instruction-aware embeddings for improved retrieval
- Similarity computation between texts
- Automatic model caching and memory optimization
"""

from .server import main, mcp_server
from .config import Settings
from .embedding_engine import EmbeddingEngine

__version__ = "1.0.0"
__all__ = ["main", "mcp_server", "Settings", "EmbeddingEngine"]
