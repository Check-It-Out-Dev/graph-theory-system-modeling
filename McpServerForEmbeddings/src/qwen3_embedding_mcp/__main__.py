"""
Qwen3-Embedding Server - Dual Mode Entry Point.

Supports both MCP mode (for Claude Desktop) and REST API mode (for Neo4j APOC).

Usage:
    # MCP mode (default, for Claude Desktop)
    python -m qwen3_embedding_mcp
    
    # REST API mode (for Neo4j APOC direct integration)
    python -m qwen3_embedding_mcp --rest
    python -m qwen3_embedding_mcp --rest --port 7999 --host 0.0.0.0
    
    # With debug logging
    python -m qwen3_embedding_mcp --rest --debug
    
    # Skip model preloading (faster startup, slower first request)
    python -m qwen3_embedding_mcp --rest --no-preload

Architecture:
    MCP Mode:
        Claude Desktop ──MCP──> server.py ──> embedding_engine.py ──> Qwen3-8B
        
    REST Mode (bypasses Claude context):
        Neo4j APOC ──HTTP──> rest_server.py ──> embedding_engine.py ──> Qwen3-8B
        
    The REST API is OpenAI-compatible and uses Information Lensing via
    the 'model' parameter (semantic, structural, behavioral).
"""

import argparse
import logging
import sys


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Qwen3-Embedding Server - MCP and REST API modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  MCP mode (for Claude Desktop):
    python -m qwen3_embedding_mcp

  REST API mode (for Neo4j APOC):
    python -m qwen3_embedding_mcp --rest
    python -m qwen3_embedding_mcp --rest --port 9000
    
  Neo4j APOC usage:
    CALL apoc.ml.openai.embedding(['text'], 'x', {
        endpoint: 'http://localhost:7999/v1',
        model: 'semantic'
    })
"""
    )
    
    # Mode selection
    parser.add_argument(
        "--rest",
        action="store_true",
        help="Run as REST API server instead of MCP server (for Neo4j APOC)"
    )
    
    # REST API configuration
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="REST API host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7999,
        help="REST API port (default: 7999)"
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip model preloading at startup (faster start, slower first request)"
    )
    
    # Logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Set logging level explicitly"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = args.log_level or ("DEBUG" if args.debug else "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,  # MCP uses stdout for protocol, so log to stderr
    )
    
    logger = logging.getLogger(__name__)
    
    if args.rest:
        # REST API mode for Neo4j APOC
        logger.info("Starting in REST API mode")
        from .rest_server import run_rest_server
        run_rest_server(
            host=args.host,
            port=args.port,
            debug=args.debug,
            preload_model=not args.no_preload
        )
    else:
        # MCP mode for Claude Desktop
        logger.info("Starting in MCP server mode")
        from .server import main as mcp_main
        mcp_main()


if __name__ == "__main__":
    main()
