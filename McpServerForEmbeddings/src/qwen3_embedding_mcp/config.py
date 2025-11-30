"""
Configuration management for Qwen3-Embedding MCP Server.

Uses Pydantic Settings for type-safe configuration with environment variable support.
Includes Information Lensing domain instructions for triple embedding generation.
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# =============================================================================
# Information Lensing Domain Instructions
# =============================================================================
# These act as "prompt-based gravitational lenses" that curve the embedding space
# toward domain-relevant semantic structure without requiring matrix transformations.
#
# Reference: Information Lensing Framework (Marchewka, 2025)
# See: appendix_information_lensing.md, enhanced_graph_pipeline.md

DOMAIN_INSTRUCTIONS: dict[str, str] = {
    # -------------------------------------------------------------------------
    # STRUCTURAL LENS: Graph topology, architectural position, connectivity
    # -------------------------------------------------------------------------
    "structural": (
        "Embed the STRUCTURAL TOPOLOGY of code in a directed heterogeneous hypergraph. "
        "Focus ONLY on: "
        "graph connectivity (in-degree, out-degree, directed paths), "
        "centrality measures (betweenness, pagerank, eigenvector), "
        "community structure and clustering, "
        "node types (Controller, Service, Repository, Entity, Config), "
        "edge types (METHOD_CALL, DEPENDENCY_INJECTION, DATA_FLOW), "
        "hyperedge participation with source/target roles, "
        "design patterns and architectural motifs. "
        "Completely IGNORE what the code does - only HOW it's connected."
    ),
    
    # -------------------------------------------------------------------------
    # SEMANTIC LENS: Business domain, intent, conceptual meaning
    # -------------------------------------------------------------------------
    "semantic": (
        "Embed the SEMANTIC MEANING of Spring Boot code for CheckItOut. "
        "Focus ONLY on: "
        "business logic (influencer marketing, campaigns, payments), "
        "what this code DOES functionally, "
        "algorithms and data transformations, "
        "domain-specific terminology, "
        "API contracts and interfaces. "
        "Completely IGNORE structure and runtime - only WHAT it means."
    ),
    
    # -------------------------------------------------------------------------
    # BEHAVIORAL LENS: Runtime patterns, execution flow, side effects
    # -------------------------------------------------------------------------
    "behavioral": (
        "Embed the RUNTIME BEHAVIOR of code execution. "
        "Focus ONLY on: "
        "state machines and transitions, "
        "error handling and recovery patterns, "
        "retry logic and circuit breakers, "
        "transaction boundaries, "
        "async operations and threading, "
        "side effects (DB writes, network calls, events), "
        "causal relationships and downstream effects. "
        "Completely IGNORE static structure and meaning - only HOW it behaves."
    ),
}

# Valid lens types for type checking
LensType = Literal["structural", "semantic", "behavioral"]


class Settings(BaseSettings):
    """
    Server configuration settings.
    
    All settings can be configured via environment variables with the
    QWEN3_EMBEDDING_ prefix. For example:
    - QWEN3_EMBEDDING_MODEL_ID=Qwen/Qwen3-Embedding-8B
    - QWEN3_EMBEDDING_DEVICE=cpu
    """
    
    model_config = SettingsConfigDict(
        env_prefix="QWEN3_EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Model settings
    model_id: str = Field(
        default="Qwen/Qwen3-Embedding-8B",
        description="HuggingFace model ID or local path to the embedding model",
    )
    
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="cpu",
        description="Device to run inference on (cpu, cuda, mps, auto)",
    )
    
    torch_dtype: Literal["float32", "float16", "bfloat16", "auto"] = Field(
        default="float32",
        description="PyTorch dtype for model weights",
    )
    
    # Embedding settings
    default_dimension: int = Field(
        default=4096,
        ge=128,
        le=4096,
        description="Default embedding dimension (Qwen3-8B supports up to 4096)",
    )
    
    max_sequence_length: int = Field(
        default=32768,
        ge=512,
        le=32768,
        description="Maximum input sequence length in tokens (Qwen3-8B supports 32K)",
    )
    
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to L2-normalize embeddings by default",
    )
    
    # Performance settings
    batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Maximum batch size for embedding generation",
    )
    
    show_progress: bool = Field(
        default=False,
        description="Show progress bar during batch processing",
    )
    
    # Server settings
    server_name: str = Field(
        default="qwen3-embedding-mcp",
        description="MCP server name for identification",
    )
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )
    
    # Cache settings
    cache_dir: str | None = Field(
        default=None,
        description="Custom cache directory for model weights (defaults to HuggingFace cache)",
    )
    
    trust_remote_code: bool = Field(
        default=True,
        description="Whether to trust remote code from HuggingFace Hub",
    )


# Global settings instance
settings = Settings()
