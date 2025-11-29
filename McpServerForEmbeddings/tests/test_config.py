"""
Tests for configuration module.
"""

import os
import pytest
from qwen3_embedding_mcp.config import Settings


def test_default_settings():
    """Test default settings values."""
    settings = Settings()
    
    assert settings.model_id == "Qwen/Qwen3-Embedding-8B"
    assert settings.device == "cpu"
    assert settings.torch_dtype == "float32"
    assert settings.default_dimension == 4096
    assert settings.normalize_embeddings is True


def test_settings_from_env(monkeypatch):
    """Test settings can be configured via environment variables."""
    monkeypatch.setenv("QWEN3_EMBEDDING_MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.setenv("QWEN3_EMBEDDING_DEVICE", "cuda")
    monkeypatch.setenv("QWEN3_EMBEDDING_DEFAULT_DIMENSION", "1024")
    
    settings = Settings()
    
    assert settings.model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.device == "cuda"
    assert settings.default_dimension == 1024


def test_dimension_validation():
    """Test dimension bounds validation."""
    # Valid dimensions
    settings = Settings(default_dimension=128)
    assert settings.default_dimension == 128
    
    settings = Settings(default_dimension=4096)
    assert settings.default_dimension == 4096
    
    # Invalid dimensions should raise
    with pytest.raises(ValueError):
        Settings(default_dimension=64)  # Too small
    
    with pytest.raises(ValueError):
        Settings(default_dimension=8192)  # Too large
