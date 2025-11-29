"""
Information Lensing Reranker MCP Server

Domain-tuned reranker for semantic similarity scoring in enterprise codebases.

Simple interface:
    score_pair(code_a, code_b) → probability (0.0-1.0)

The server uses custom instructions that transform generic relevance scoring
into semantic domain similarity scoring - detecting when structurally similar
code belongs to different business domains.

Example:
    >>> from qwen3_reranker_mcp import get_engine
    >>> 
    >>> engine = get_engine()
    >>> result = engine.score_pair(payment_service, inventory_service)
    >>> print(f"Score: {result.score:.3f}")  # 0.08 - different domains
    >>> 
    >>> # Compare with embedding similarity for Information Lensing
    >>> embedding_sim = 0.94
    >>> divergence = embedding_sim - result.score  # 0.86 - needs lens correction!

See Also:
    - appendix_information_lensing.md: Theory behind the approach
"""

from .config import Settings, get_settings
from .reranker_engine import RerankerEngine, ScoreResult, get_engine
from .server import main, mcp_server

__version__ = "2.0.0"
__author__ = "Norbert Marchewka"

__all__ = [
    # Server
    "main",
    "mcp_server",
    # Configuration
    "Settings",
    "get_settings",
    # Engine
    "RerankerEngine",
    "get_engine",
    # Result type
    "ScoreResult",
]
