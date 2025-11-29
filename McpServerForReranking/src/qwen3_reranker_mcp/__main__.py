"""
Entry point for running the MCP server as a module.

Usage:
    python -m qwen3_reranker_mcp
    
Or via the installed entry point:
    qwen3-reranker-mcp
"""

from .server import main

if __name__ == "__main__":
    main()
