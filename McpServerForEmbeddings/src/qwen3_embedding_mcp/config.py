"""
Configuration management for Qwen3-Embedding MCP Server.

Uses Pydantic Settings for type-safe configuration with environment variable support.
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


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
