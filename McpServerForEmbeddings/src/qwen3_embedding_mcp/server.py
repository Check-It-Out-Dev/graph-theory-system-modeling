"""
Qwen3-Embedding MCP Server Implementation.

Provides embedding generation capabilities via the Model Context Protocol.
Designed for integration with Claude Desktop, Cursor, and other MCP clients.
"""

import asyncio
import logging
from typing import Annotated

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)
from pydantic import BaseModel, Field

from .config import settings
from .embedding_engine import EmbeddingEngine, get_engine

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp_server = Server(settings.server_name)


# ============================================================================
# Tool Input Schemas
# ============================================================================

class EmbedInput(BaseModel):
    """Input schema for the embed tool."""
    
    texts: list[str] | str = Field(
        description="Text or list of texts to generate embeddings for. "
                   "Maximum 32K tokens per text."
    )
    instruction: str | None = Field(
        default=None,
        description="Custom instruction for task-specific embeddings. "
                   "Example: 'Given a scientific question, retrieve relevant papers'. "
                   "If provided, overrides prompt_name. Use for specialized tasks."
    )
    prompt_name: str | None = Field(
        default=None,
        description="Built-in prompt for retrieval tasks: 'query' or 'passage'. "
                   "Use 'query' for search queries (adds retrieval instruction). "
                   "Use 'passage' for documents being indexed. "
                   "Recommended over custom instruction for standard retrieval."
    )
    is_query: bool = Field(
        default=False,
        description="Shorthand for prompt_name='query'. "
                   "Set to true when embedding search queries."
    )
    dimension: int | None = Field(
        default=None,
        ge=128,
        le=4096,
        description="Output embedding dimension. Smaller dimensions are faster but less accurate. "
                   "Supports Matryoshka Representation Learning (MRL) for flexible dimensions. "
                   "Default: 4096 (full dimension)."
    )
    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize embeddings. "
                   "Normalized embeddings allow using dot product for similarity."
    )


class SimilarityInput(BaseModel):
    """Input schema for the similarity tool."""
    
    queries: list[str] | str = Field(
        description="Query text(s) to compare against documents."
    )
    documents: list[str] | str = Field(
        description="Document text(s) to compare against queries."
    )
    query_instruction: str | None = Field(
        default=None,
        description="Optional instruction for query embeddings. "
                   "Example: 'Given a question, retrieve passages that answer it'."
    )


class BatchEmbedInput(BaseModel):
    """Input schema for batch embedding with different instructions."""
    
    items: list[dict] = Field(
        description="List of items to embed. Each item should have 'text' and optionally "
                   "'instruction' and 'type' (query/document) fields."
    )
    dimension: int | None = Field(
        default=None,
        ge=128,
        le=4096,
        description="Output embedding dimension for all items."
    )


# ============================================================================
# Tool Definitions
# ============================================================================

TOOLS = [
    Tool(
        name="embed",
        description="""Generate high-quality text embeddings using Qwen3-Embedding-8B.

This tool converts text into dense vector representations (embeddings) that capture 
semantic meaning. These embeddings can be used for:
- Semantic search and retrieval
- Document similarity comparison  
- Clustering and classification
- RAG (Retrieval-Augmented Generation) pipelines
- Vector database storage

The model is ranked #1 on the MTEB multilingual leaderboard (score 70.58) and 
supports 100+ languages with 32K token context.

The model is INSTRUCTION-AWARE. For best retrieval results:
- For queries: Set is_query=true OR prompt_name="query" (recommended)
- For documents: Set prompt_name="passage" 
- For custom tasks: Use instruction parameter with your specific instruction

Using built-in prompts (query/passage) improves retrieval performance by 1-5%.
Smaller dimensions (512, 1024) via MRL are faster but slightly less accurate.""",
        inputSchema=EmbedInput.model_json_schema(),
    ),
    Tool(
        name="similarity",
        description="""Compute semantic similarity between queries and documents.

Returns a similarity matrix where each score represents how semantically similar
a query is to a document. Scores range from -1 to 1, where:
- 1.0 = Identical meaning
- 0.0 = Unrelated
- -1.0 = Opposite meaning (rare)

Useful for:
- Finding the most relevant documents for a query
- Reranking search results
- Duplicate detection
- Semantic matching""",
        inputSchema=SimilarityInput.model_json_schema(),
    ),
    Tool(
        name="model_info",
        description="""Get information about the loaded embedding model.

Returns details about the model including:
- Model ID and status
- Device (CPU/GPU)
- Maximum sequence length
- Embedding dimensions
- Data type

Useful for debugging and verifying the model is properly loaded.""",
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
    """
    Handle tool calls from MCP clients.
    
    Args:
        name: The tool name to execute.
        arguments: Tool-specific arguments.
        
    Returns:
        CallToolResult with the tool output.
    """
    try:
        engine = get_engine()
        
        if name == "embed":
            return await _handle_embed(engine, arguments)
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
    """Handle the embed tool call."""
    input_data = EmbedInput(**arguments)
    
    # Run embedding in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: engine.embed(
            texts=input_data.texts,
            instruction=input_data.instruction,
            prompt_name=input_data.prompt_name,
            is_query=input_data.is_query,
            dimension=input_data.dimension,
            normalize=input_data.normalize,
        )
    )
    
    # Format response
    response = {
        "embeddings": result.embeddings,
        "dimensions": result.dimensions,
        "num_texts": result.num_texts,
        "normalized": result.normalized,
        "model": result.model_id,
    }
    
    # Create summary for Claude
    import json
    summary = (
        f"Generated {result.num_texts} embedding(s) with {result.dimensions} dimensions.\n\n"
        f"```json\n{json.dumps(response, indent=2)}\n```"
    )
    
    return CallToolResult(
        content=[TextContent(type="text", text=summary)]
    )


async def _handle_similarity(engine: EmbeddingEngine, arguments: dict) -> CallToolResult:
    """Handle the similarity tool call."""
    input_data = SimilarityInput(**arguments)
    
    # Run similarity computation in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: engine.similarity(
            queries=input_data.queries,
            documents=input_data.documents,
            query_instruction=input_data.query_instruction,
        )
    )
    
    # Format response with ranked results
    import json
    
    queries = input_data.queries if isinstance(input_data.queries, list) else [input_data.queries]
    documents = input_data.documents if isinstance(input_data.documents, list) else [input_data.documents]
    
    # Create ranked results for each query
    ranked_results = []
    for i, query in enumerate(queries):
        scores = result.scores[i]
        ranked = sorted(
            [(j, documents[j], scores[j]) for j in range(len(documents))],
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
    
    summary = f"Computed similarity for {result.num_queries} query(ies) against {result.num_documents} document(s).\n\n"
    summary += f"```json\n{json.dumps(response, indent=2)}\n```"
    
    return CallToolResult(
        content=[TextContent(type="text", text=summary)]
    )


async def _handle_model_info(engine: EmbeddingEngine) -> CallToolResult:
    """Handle the model_info tool call."""
    import json
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
    # Configure logging
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
