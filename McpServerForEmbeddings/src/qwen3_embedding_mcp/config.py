"""
Configuration management for Qwen3-Embedding MCP Server.

Uses Pydantic Settings for type-safe configuration with environment variable support.
Includes Information Lensing domain instructions for triple embedding generation.
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import hashlib
from datetime import datetime


# =============================================================================
# Embedding Version Constants
# =============================================================================
# Version tracking for embeddings ensures:
# 1. We can detect when lens instructions change (embedding invalidation)
# 2. We can track model versions across updates
# 3. NavigationMaster can store which version was used for each embedding

EMBEDDING_SERVER_VERSION = "2.0.1"  # Server version
MODEL_VERSION = "2025.11.30-v1"     # Model version (update when model changes)


def compute_lens_hash(lens_instruction: str) -> str:
    """
    Compute a short hash of lens instruction for version tracking.

    This allows detecting when lens instructions change, which would
    require re-generating embeddings for affected files.

    Returns first 8 characters of SHA-256 hash.
    """
    return hashlib.sha256(lens_instruction.encode('utf-8')).hexdigest()[:8]


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


# =============================================================================
# Computed Lens Hashes (auto-computed from instructions above)
# =============================================================================
# These are used to detect when lens instructions change, signaling
# that embeddings generated with old instructions should be invalidated.

LENS_HASHES: dict[str, str] = {
    lens: compute_lens_hash(instruction)
    for lens, instruction in DOMAIN_INSTRUCTIONS.items()
}


def get_embedding_version_metadata() -> dict:
    """
    Get complete embedding version metadata for storage in Neo4j.

    Returns a dictionary suitable for JSON serialization and storage
    in EntityDetail.embedding_version property.

    Example output:
    {
        "model_id": "Qwen/Qwen3-Embedding-8B",
        "model_version": "2025.11.30-v1",
        "server_version": "2.0.1",
        "lens_versions": {
            "semantic": "a1b2c3d4",
            "behavioral": "e5f6g7h8",
            "structural": "i9j0k1l2"
        },
        "generated_at": "2025-11-30T12:00:00Z"
    }
    """
    return {
        "model_id": settings.model_id if 'settings' in dir() else "Qwen/Qwen3-Embedding-8B",
        "model_version": MODEL_VERSION,
        "server_version": EMBEDDING_SERVER_VERSION,
        "lens_versions": LENS_HASHES.copy(),
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }


def check_lens_version_match(stored_hash: str, lens: LensType) -> bool:
    """
    Check if a stored lens hash matches the current lens instruction.

    Use this to detect if an embedding needs regeneration due to
    lens instruction changes.

    Args:
        stored_hash: Hash stored with the embedding in Neo4j
        lens: Which lens to check against

    Returns:
        True if hashes match (embedding still valid)
        False if hashes differ (embedding should be regenerated)
    """
    return stored_hash == LENS_HASHES.get(lens)


class Settings(BaseSettings):
    """
    Server configuration settings.
    
    All settings can be configured via environment variables with the
    QWEN3_EMBEDDING_ prefix. For example:
    - QWEN3_EMBEDDING_MODEL_ID=Qwen/Qwen3-Embedding-8B
    - QWEN3_EMBEDDING_DEVICE=cpu
    - QWEN3_EMBEDDING_REST_PORT=8080
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
    
    # MCP Server settings
    server_name: str = Field(
        default="qwen3-embedding-mcp",
        description="MCP server name for identification",
    )
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )
    
    # REST API settings (for Neo4j APOC integration)
    rest_host: str = Field(
        default="0.0.0.0",
        description="Host to bind REST API server (0.0.0.0 for all interfaces)",
    )
    
    rest_port: int = Field(
        default=7999,
        ge=1,
        le=65535,
        description="Port for REST API server",
    )
    
    rest_debug: bool = Field(
        default=False,
        description="Enable Flask debug mode (not for production)",
    )
    
    rest_preload_model: bool = Field(
        default=True,
        description="Preload model at REST server startup for faster first request",
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
