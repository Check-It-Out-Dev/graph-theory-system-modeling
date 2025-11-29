"""
Information Lensing Reranker MCP Server.

Simple interface for semantic domain similarity scoring.

Tools:
- score_pair: Input two code segments → Output probability score (0.0-1.0)
- model_info: Get model status

The server is tuned for enterprise Java/Angular/DevOps codebases.
It uses custom instructions that shift from "relevance" to "semantic domain similarity".

Usage with Claude:
    "Score the semantic similarity between these two files"
    → Returns probability that they belong to the same semantic domain

Integration with Information Lensing:
    divergence = embedding_similarity - reranker_score
    if divergence > 0.4:
        # This pair needs lens correction
"""

import asyncio
import json
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import BaseModel, Field

from .config import get_settings
from .reranker_engine import get_engine

logger = logging.getLogger(__name__)
settings = get_settings()
mcp_server = Server(settings.server_name)


# =============================================================================
# Input Schema
# =============================================================================

class ScorePairInput(BaseModel):
    """Input: two code segments to compare."""
    
    query: str = Field(
        description="First code segment (file content, code snippet, or config)"
    )
    document: str = Field(
        description="Second code segment to compare against the first"
    )
    symmetric: bool = Field(
        default=True,
        description="If True (default), compute (score(a,b) + score(b,a)) / 2 for symmetric similarity"
    )


class ScoreBatchInput(BaseModel):
    """Input: multiple pairs to score in one call."""
    
    pairs: list[tuple[str, str]] = Field(
        description="List of (code_a, code_b) pairs to score",
        min_length=1,
        max_length=100,
    )
    symmetric: bool = Field(
        default=True,
        description="If True, compute (score(a,b) + score(b,a)) / 2 for each pair"
    )


# =============================================================================
# Tools
# =============================================================================

TOOLS = [
    Tool(
        name="score_pair",
        description="""Score semantic domain similarity between two code segments.

Returns probability (0.0-1.0) that the segments belong to the SAME semantic domain.

Tuned for enterprise codebases:
- Java Spring (Services, Controllers, Repositories, Entities)
- Angular (Components, Services, Modules, Guards)
- Ansible (Playbooks, Roles, Tasks)
- CI/CD (Pipelines, Workflows)
- Configs (YAML, JSON, properties)

Scoring:
- HIGH (0.7-1.0): Same business domain (both handle payments)
- MEDIUM (0.3-0.7): Related (PaymentService ↔ PaymentController)
- LOW (0.0-0.3): Different domains despite structural similarity

For Information Lensing:
    divergence = embedding_sim - this_score
    High divergence (>0.4) = lens needs to correct this pair""",
        inputSchema=ScorePairInput.model_json_schema(),
    ),
    Tool(
        name="score_batch",
        description="""Score multiple pairs in a single call. Optimized for building similarity matrices.

Input: List of (code_a, code_b) pairs (max 100)
Output: List of scores in same order

By default uses symmetric scoring: (score(a,b) + score(b,a)) / 2
This ensures S[i,j] = S[j,i] as required for Information Lensing.

Example usage for matrix construction:
    pairs = [(files[i], files[j]) for i in range(N) for j in range(i+1, N)]
    result = score_batch(pairs, symmetric=True)
    # result.scores contains upper triangular matrix values""",
        inputSchema=ScoreBatchInput.model_json_schema(),
    ),
    Tool(
        name="model_info",
        description="Get reranker model status and configuration.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


# =============================================================================
# Handlers
# =============================================================================

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    try:
        engine = get_engine()
        
        if name == "score_pair":
            input_data = ScorePairInput(**arguments)
            
            # Run scoring in thread pool
            loop = asyncio.get_event_loop()
            
            if input_data.symmetric:
                # Symmetric scoring: (score(a,b) + score(b,a)) / 2
                score = await loop.run_in_executor(
                    None,
                    lambda: engine.score_pair_symmetric(input_data.query, input_data.document),
                )
            else:
                # Asymmetric scoring: query → document only
                result = await loop.run_in_executor(
                    None,
                    lambda: engine.score_pair(input_data.query, input_data.document),
                )
                score = result.score
            
            # Simple response: just the score and minimal context
            response = {
                "score": round(score, 4),
                "interpretation": _interpret_score(score),
                "symmetric": input_data.symmetric,
            }
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"**Score: {score:.4f}**\n\n```json\n{json.dumps(response, indent=2)}\n```"
                )]
            )
        
        elif name == "score_batch":
            input_data = ScoreBatchInput(**arguments)
            
            # Convert list of lists to list of tuples (JSON doesn't have tuples)
            pairs = [tuple(p) for p in input_data.pairs]
            
            # Run batch scoring in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: engine.score_batch(pairs, symmetric=input_data.symmetric),
            )
            
            # Response with all scores
            response = {
                "scores": [round(s, 4) for s in result.scores],
                "count": len(result.scores),
                "symmetric": result.symmetric,
            }
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"**Batch complete: {len(result.scores)} pairs scored**\n\n```json\n{json.dumps(response, indent=2)}\n```"
                )]
            )
        
        elif name == "model_info":
            info = engine.get_model_info()
            return CallToolResult(
                content=[TextContent(type="text", text=f"```json\n{json.dumps(info, indent=2)}\n```")]
            )
        
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )
            
    except Exception as e:
        logger.exception(f"Error in {name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {e}")],
            isError=True,
        )


def _interpret_score(score: float) -> str:
    """Simple interpretation of the score."""
    if score >= 0.7:
        return "HIGH - Same semantic domain"
    elif score >= 0.3:
        return "MEDIUM - Related domains"
    else:
        return "LOW - Different domains (structural similarity only)"


# =============================================================================
# Server Lifecycle
# =============================================================================

async def run_server() -> None:
    """Run the MCP server."""
    logger.info(f"Starting {settings.server_name}")
    logger.info(f"Model: {settings.model_id}")
    logger.info(f"Custom instruction: {'ENABLED' if settings.use_custom_instruction else 'DISABLED'}")
    
    # Pre-load model
    engine = get_engine()
    if not engine.is_loaded:
        logger.info("Loading model...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: engine.model)
        await loop.run_in_executor(None, engine.warmup)
    
    logger.info("Server ready")
    
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Shutdown")


if __name__ == "__main__":
    main()
