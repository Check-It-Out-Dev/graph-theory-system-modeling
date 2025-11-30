"""
Qwen3-Embedding MCP Server with Information Lensing.

Provides embedding generation via the Model Context Protocol with three
domain-specific "gravitational lenses" for code understanding:
- structural: Graph topology, architectural position
- semantic: Business domain, code meaning
- behavioral: Runtime patterns, execution flow

Reference: Marchewka (2025), "Information Lensing: A Gravitational 
Approach to Domain-Specific Embedding Transformation"
"""

import asyncio
import json
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)
from pydantic import BaseModel, Field

from .config import settings, DOMAIN_INSTRUCTIONS, LensType
from .embedding_engine import EmbeddingEngine, get_engine

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp_server = Server(settings.server_name)


# =============================================================================
# Tool Input Schemas (Simplified)
# =============================================================================

class EmbedInput(BaseModel):
    """Input schema for single text embedding with Information Lensing."""
    
    text: str = Field(
        description="Document content to embed (max 32K tokens)"
    )
    lens: LensType = Field(
        description="Information Lens type: 'structural', 'semantic', or 'behavioral'"
    )
    dimension: int | None = Field(
        default=None,
        ge=128,
        le=4096,
        description="Output dimension (128-4096). Default: 4096"
    )


class BatchEmbedInput(BaseModel):
    """Input schema for batch embedding with same lens."""
    
    texts: list[str] = Field(
        description="List of documents to embed (max 20)",
        min_length=1,
        max_length=20,
    )
    lens: LensType = Field(
        description="Information Lens type: 'structural', 'semantic', or 'behavioral'"
    )
    dimension: int | None = Field(
        default=None,
        ge=128,
        le=4096,
        description="Output dimension. Default: 4096"
    )


# =============================================================================
# Tool Definitions
# =============================================================================

TOOLS = [
    Tool(
        name="embed",
        description="""Generate embedding for a document using Information Lensing.

Apply one of three "gravitational lenses" to focus the embedding:
- structural: Graph topology, architectural position, connectivity
- semantic: Business logic, domain concepts, code meaning
- behavioral: Runtime patterns, state machines, side effects

Returns 4096-dimensional vector (or custom dimension via MRL).

Example: embed("class PaymentService...", lens="semantic")""",
        inputSchema=EmbedInput.model_json_schema(),
    ),
    Tool(
        name="batch_embed",
        description="""Generate embeddings for multiple documents (max 20) with same lens.

More efficient than calling embed() multiple times.
All documents get the same lens applied.""",
        inputSchema=BatchEmbedInput.model_json_schema(),
    ),
    Tool(
        name="model_info",
        description="""Get model status and available lenses.""",
        inputSchema={"type": "object", "properties": {}},
    ),
]


# =============================================================================
# Tool Handlers
# =============================================================================

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
    """Handle single text embedding with lens."""
    input_data = EmbedInput(**arguments)
    
    logger.info(f"Embedding with {input_data.lens} lens ({len(input_data.text)} chars)")
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: engine.embed(
            texts=input_data.text,
            lens=input_data.lens,
            dimension=input_data.dimension,
        )
    )
    
    response = {
        "embedding": result.embeddings[0],
        "lens": result.lens,
        "dimensions": result.dimensions,
        "normalized": result.normalized,
    }
    
    return CallToolResult(
        content=[TextContent(
            type="text", 
            text=f"Generated {input_data.lens} embedding ({result.dimensions}D)\n\n```json\n{json.dumps(response, indent=2)}\n```"
        )]
    )


async def _handle_batch_embed(engine: EmbeddingEngine, arguments: dict) -> CallToolResult:
    """Handle batch embedding with same lens."""
    input_data = BatchEmbedInput(**arguments)
    
    logger.info(f"Batch embedding {len(input_data.texts)} texts with {input_data.lens} lens")
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: engine.embed(
            texts=input_data.texts,
            lens=input_data.lens,
            dimension=input_data.dimension,
        )
    )
    
    response = {
        "embeddings": result.embeddings,
        "lens": result.lens,
        "num_texts": result.num_texts,
        "dimensions": result.dimensions,
    }
    
    return CallToolResult(
        content=[TextContent(
            type="text",
            text=f"Generated {result.num_texts} {input_data.lens} embeddings ({result.dimensions}D)\n\n```json\n{json.dumps(response, indent=2)}\n```"
        )]
    )


async def _handle_model_info(engine: EmbeddingEngine) -> CallToolResult:
    """Handle model info request."""
    info = engine.get_model_info()
    
    # Add lens descriptions
    info["lens_descriptions"] = {
        lens: instr[:100] + "..."
        for lens, instr in DOMAIN_INSTRUCTIONS.items()
    }
    
    return CallToolResult(
        content=[TextContent(type="text", text=f"```json\n{json.dumps(info, indent=2)}\n```")]
    )


# =============================================================================
# Server Lifecycle
# =============================================================================

async def run_server() -> None:
    """Run the MCP server using stdio transport."""
    logger.info(f"Starting {settings.server_name} v2.0.0 (Information Lensing)")
    logger.info(f"Model: {settings.model_id}")
    logger.info(f"Available lenses: {list(DOMAIN_INSTRUCTIONS.keys())}")
    
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
