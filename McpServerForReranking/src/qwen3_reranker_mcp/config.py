"""
Configuration for Enterprise Domain-Tuned Reranker.

Optimized for Information Lensing in Java Spring / Angular / DevOps codebases.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Domain-Specific System Instruction
# =============================================================================

# This instruction transforms the reranker from "relevance detector" to
# "semantic domain similarity detector" - the key shift for Information Lensing

INFORMATION_LENSING_INSTRUCTION = """You are evaluating semantic domain similarity between two code segments.

TASK: Determine probability (0.0-1.0) that these segments belong to the SAME semantic domain and would change together during development.

SCORING RULES:
- HIGH (0.7-1.0): Same business domain (both handle payments, both handle users, etc.)
- MEDIUM (0.3-0.7): Related domains (PaymentService ↔ PaymentController)  
- LOW (0.0-0.3): Different domains despite structural similarity

CRITICAL - IGNORE STRUCTURAL PATTERNS:
For Java Spring: Ignore @Service/@Repository/@Controller annotations, autowiring patterns, similar method signatures
For Angular: Ignore @Component/@Injectable decorators, similar DI patterns
For Ansible/CI-CD: Ignore YAML structure, similar task patterns
For Configs: Ignore file format similarities

FOCUS ON:
1. Business entity/process being handled (Payment? User? Order? Inventory?)
2. Would a product requirement change affect BOTH segments?
3. Data flow dependencies between segments

Example: PaymentService.java and InventoryService.java may look 90% structurally similar but should score < 0.2 because they handle DIFFERENT business domains."""


class Settings(BaseSettings):
    """
    Server configuration optimized for Information Lensing.
    
    Key changes from generic reranker:
    - Custom instruction for domain-aware scoring
    - Tuned for enterprise Java/Angular/DevOps patterns
    """
    
    model_config = SettingsConfigDict(
        env_prefix="QWEN3_RERANKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # =========================================================================
    # Model Settings
    # =========================================================================
    
    model_id: str = Field(
        default="Qwen/Qwen3-Reranker-8B",
        description="HuggingFace model ID for the reranker.",
    )
    
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="cpu",
        description="Device for inference. 'cpu' recommended for 192GB RAM setup.",
    )
    
    torch_dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="float32",  # Full precision for maximum quality
        description="PyTorch dtype. float32 for maximum precision (recommended with 192GB RAM).",
    )
    
    # =========================================================================
    # Information Lensing Settings
    # =========================================================================
    
    custom_instruction: str = Field(
        default=INFORMATION_LENSING_INSTRUCTION,
        description="System instruction for domain-aware scoring.",
    )
    
    use_custom_instruction: bool = Field(
        default=True,
        description="Whether to use custom instruction for scoring.",
    )
    
    # =========================================================================
    # Inference Settings
    # =========================================================================
    
    max_length: int = Field(
        default=8192,
        ge=512,
        le=32768,
        description="Maximum sequence length for code pairs.",
    )
    
    batch_size: int = Field(
        default=1,  # Single pair focus
        ge=1,
        le=64,
        description="Batch size (1 for iterative pair-by-pair scoring).",
    )
    
    default_top_k: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Default top-k for reranking (not primary use case).",
    )
    
    # =========================================================================
    # Server Settings
    # =========================================================================
    
    server_name: str = Field(
        default="information-lensing-reranker",
        description="MCP server name.",
    )
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity.",
    )
    
    # =========================================================================
    # Cache Settings
    # =========================================================================
    
    cache_dir: str | None = Field(
        default=None,
        description="Custom cache directory for model weights.",
    )
    
    trust_remote_code: bool = Field(
        default=True,
        description="Required for Qwen models.",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
