"""
Qwen3-Embedding MCP Server Implementation.

Provides embedding generation capabilities via the Model Context Protocol.
Designed for integration with Claude Desktop, Cursor, and other MCP clients.
"""

import asyncio
import json
import logging
from typing import Annotated

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)
from pydantic import BaseModel, Field, field_validator

from .config import settings
from .embedding_engine import EmbeddingEngine, get_engine

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp_server = Server(settings.server_name)


# ============================================================================
# Tool Input Schemas
# ============================================================================

class EmbedInput(BaseModel):
    """Input schema for single text embedding."""
    
    text: str = Field(
        description="Single text to generate embedding for. Maximum 32K tokens."
    )
    instruction: str | None = Field(
        default=None,
        description="Custom instruction for task-specific embeddings. "
                   "Example: 'Given a scientific question, retrieve relevant papers'. "
                   "If provided, overrides prompt_name."
    )
    prompt_name: str | None = Field(
        default=None,
        description="Built-in prompt: 'query' (for search queries) or 'document' (for indexing)."
    )
    is_query: bool = Field(
        default=False,
        description="Shorthand for prompt_name='query'. Set true for search queries."
    )
    dimension: int | None = Field(
        default=None,
        ge=128,
        le=4096,
        description="Output dimension (128-4096). Smaller = faster but less accurate. Default: 4096."
    )
    normalize: bool = Field(
        default=True,
        description="L2-normalize embeddings. Enables dot product for similarity."
    )


class BatchEmbedInput(BaseModel):
    """Input schema for batch text embedding (max 20 texts)."""
    
    texts: list[str] = Field(
        description="List of texts to embed. Maximum 20 texts, each up to 32K tokens.",
        min_length=1,
        max_length=20,
    )
    instruction: str | None = Field(
        default=None,
        description="Custom instruction applied to all texts."
    )
    prompt_name: str | None = Field(
        default=None,
        description="Built-in prompt: 'query' or 'document'. Applied to all texts."
    )
    is_query: bool = Field(
        default=False,
        description="Shorthand for prompt_name='query'."
    )
    dimension: int | None = Field(
        default=None,
        ge=128,
        le=4096,
        description="Output dimension for all embeddings. Default: 4096."
    )
    normalize: bool = Field(
        default=True,
        description="L2-normalize all embeddings."
    )


class SimilarityInput(BaseModel):
    """Input schema for the similarity tool."""
    
    queries: list[str] = Field(
        description="Query texts to compare (max 10).",
        min_length=1,
        max_length=10,
    )
    documents: list[str] = Field(
        description="Document texts to compare against (max 20).",
        min_length=1,
        max_length=20,
    )
    query_instruction: str | None = Field(
        default=None,
        description="Optional instruction for query embeddings."
    )


# ============================================================================
# Tool Definitions
# ============================================================================

TOOLS = [
    Tool(
        name="embed",
        description="""Generate embedding for a SINGLE text using Qwen3-Embedding-8B.

Converts text into a dense vector (4096 dimensions by default) capturing semantic meaning.
Use for: semantic search, similarity, clustering, RAG pipelines.

For multiple texts, use 'batch_embed' instead.

INSTRUCTION-AWARE - for best retrieval:
- Queries: Set is_query=true OR prompt_name="query"
- Documents: Set prompt_name="document"
- Custom: Use instruction parameter""",
        inputSchema=EmbedInput.model_json_schema(),
    ),
    Tool(
        name="batch_embed",
        description="""Generate embeddings for MULTIPLE texts (max 20) using Qwen3-Embedding-8B.

Efficiently processes up to 20 texts in a single call. Each text gets its own embedding.
Same instruction/prompt_name applies to all texts in the batch.

Returns array of embeddings in same order as input texts.

For single text, use 'embed' instead.""",
        inputSchema=BatchEmbedInput.model_json_schema(),
    ),
    Tool(
        name="similarity",
        description="""Compute semantic similarity between queries and documents.

Returns similarity matrix (queries × documents). Scores range -1 to 1:
- 1.0 = Identical meaning
- 0.0 = Unrelated  
- -1.0 = Opposite meaning

Max 10 queries × 20 documents per call.""",
        inputSchema=SimilarityInput.model_json_schema(),
    ),
    Tool(
        name="model_info",
        description="""Get information about the loaded embedding model.

Returns: model ID, device (CPU/GPU), max sequence length, embedding dimensions, dtype.""",
        inputSchema={"type": "object", "properties": {}},
    ),
]


# ============================================================================
# Tool Handlers
# ============================================================================

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available tools."""
    return TOOLS


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle tool calls from MCP clients."""
    try:
        engine = get_engine()
        
        if name == "embed":
            return await _handle_embed(engine, arguments)
        elif name == "batch_embed":
            return await _handle_batch_embed(engine, arguments)
        elif name == "similarity":
            return await _handle_similarity(engine, arguments)
        elif name == "model_info":
            return await _handle_model_info(engine)
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )
            
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def _handle_embed(engine: EmbeddingEngine, arguments: dict) -> CallToolResult:
    """Handle single text embedding."""
    input_data = EmbedInput(**arguments)
    
    logger.info(f"Embedding single text ({len(input_data.text)} chars)")
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: engine.embed(
            texts=input_data.text,  # Single string
            instruction=input_data.instruction,
            prompt_name=input_data.prompt_name,
            is_query=input_data.is_query,
            dimension=input_data.dimension,
            normalize=input_data.normalize,
        )
    )
    
    response = {
        "embedding": result.embeddings[0],  # Single embedding
        "dimensions": result.dimensions,
        "normalized": result.normalized,
        "model": result.model_id,
    }
    
    summary = f"Generated embedding with {result.dimensions} dimensions.\n\n```json\n{json.dumps(response, indent=2)}\n```"
    
    return CallToolResult(content=[TextContent(type="text", text=summary)])


async def _handle_batch_embed(engine: EmbeddingEngine, arguments: dict) -> CallToolResult:
    """Handle batch text embedding."""
    input_data = BatchEmbedInput(**arguments)
    
    logger.info(f"Batch embedding {len(input_data.texts)} texts")
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: engine.embed(
            texts=input_data.texts,  # List of strings
            instruction=input_data.instruction,
            prompt_name=input_data.prompt_name,
            is_query=input_data.is_query,
            dimension=input_data.dimension,
            normalize=input_data.normalize,
        )
    )
    
    response = {
        "embeddings": result.embeddings,  # List of embeddings
        "num_texts": result.num_texts,
        "dimensions": result.dimensions,
        "normalized": result.normalized,
        "model": result.model_id,
    }
    
    summary = f"Generated {result.num_texts} embeddings with {result.dimensions} dimensions each.\n\n```json\n{json.dumps(response, indent=2)}\n```"
    
    return CallToolResult(content=[TextContent(type="text", text=summary)])


async def _handle_similarity(engine: EmbeddingEngine, arguments: dict) -> CallToolResult:
    """Handle similarity computation."""
    input_data = SimilarityInput(**arguments)
    
    logger.info(f"Computing similarity: {len(input_data.queries)} queries × {len(input_data.documents)} documents")
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: engine.similarity(
            queries=input_data.queries,
            documents=input_data.documents,
            query_instruction=input_data.query_instruction,
        )
    )
    
    # Create ranked results for each query
    ranked_results = []
    for i, query in enumerate(input_data.queries):
        scores = result.scores[i]
        ranked = sorted(
            [(j, input_data.documents[j], scores[j]) for j in range(len(input_data.documents))],
            key=lambda x: x[2],
            reverse=True
        )
        ranked_results.append({
            "query": query[:100] + "..." if len(query) > 100 else query,
            "results": [
                {"rank": r + 1, "score": round(score, 4), "document": doc[:100] + "..." if len(doc) > 100 else doc}
                for r, (idx, doc, score) in enumerate(ranked)
            ]
        })
    
    response = {
        "similarity_matrix": result.scores,
        "ranked_results": ranked_results,
        "num_queries": result.num_queries,
        "num_documents": result.num_documents,
    }
    
    summary = f"Computed similarity: {result.num_queries} queries × {result.num_documents} documents.\n\n```json\n{json.dumps(response, indent=2)}\n```"
    
    return CallToolResult(content=[TextContent(type="text", text=summary)])


async def _handle_model_info(engine: EmbeddingEngine) -> CallToolResult:
    """Handle model info request."""
    info = engine.get_model_info()
    return CallToolResult(
        content=[TextContent(type="text", text=f"```json\n{json.dumps(info, indent=2)}\n```")]
    )


# ============================================================================
# Server Lifecycle
# ============================================================================

async def run_server() -> None:
    """Run the MCP server using stdio transport."""
    logger.info(f"Starting {settings.server_name} v1.0.0")
    logger.info(f"Model: {settings.model_id}")
    logger.info(f"Device: {settings.device}")
    
    # Pre-load model for faster first request
    engine = get_engine()
    if not engine.is_loaded:
        logger.info("Pre-loading embedding model (this may take 1-3 minutes)...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: engine.model)
        await loop.run_in_executor(None, engine.warmup)
    
    logger.info("Server ready, waiting for connections...")
    
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )


def main() -> None:
    """Main entry point for the MCP server."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
