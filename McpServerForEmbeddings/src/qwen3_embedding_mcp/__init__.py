"""
Qwen3-Embedding-8B MCP & REST API Server

A high-performance server providing state-of-the-art text embeddings using 
Qwen3-Embedding-8B model with Information Lensing support.

Supports two modes:
- MCP Mode: For Claude Desktop integration via Model Context Protocol
- REST API Mode: For Neo4j APOC integration via OpenAI-compatible endpoints

Features:
- Single text and batch embedding generation
- Information Lensing (semantic, structural, behavioral)
- Configurable embedding dimensions (Matryoshka Representation Learning)
- Instruction-aware embeddings for domain-specific retrieval
- OpenAI-compatible REST API for Neo4j APOC

Usage:
    # MCP mode (for Claude Desktop)
    python -m qwen3_embedding_mcp
    
    # REST API mode (for Neo4j APOC)
    python -m qwen3_embedding_mcp --rest
"""

from .server import main, mcp_server
from .config import Settings, DOMAIN_INSTRUCTIONS, LensType
from .embedding_engine import EmbeddingEngine, get_engine

# REST server is imported lazily to avoid Flask import when using MCP mode
# from .rest_server import app, run_rest_server

__version__ = "2.0.0"
__all__ = [
    "main", 
    "mcp_server", 
    "Settings", 
    "EmbeddingEngine",
    "get_engine",
    "DOMAIN_INSTRUCTIONS",
    "LensType",
]
