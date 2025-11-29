"""
Tests for embedding engine module.

Note: These tests use mocking to avoid loading the actual model,
which would require ~32GB RAM and take several minutes.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from qwen3_embedding_mcp.config import Settings
from qwen3_embedding_mcp.embedding_engine import EmbeddingEngine, EmbeddingResult


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer."""
    mock = MagicMock()
    mock.encode.return_value = np.random.randn(2, 4096).astype(np.float32)
    mock.get_sentence_embedding_dimension.return_value = 4096
    mock.max_seq_length = 8192
    mock.device = "cpu"
    return mock


@pytest.fixture
def engine_with_mock(mock_sentence_transformer):
    """Create an EmbeddingEngine with mocked model."""
    with patch("qwen3_embedding_mcp.embedding_engine.SentenceTransformer") as mock_cls:
        mock_cls.return_value = mock_sentence_transformer
        
        config = Settings(model_id="test-model")
        engine = EmbeddingEngine(config)
        engine._load_model()
        
        yield engine, mock_sentence_transformer


class TestEmbeddingEngine:
    """Tests for EmbeddingEngine class."""
    
    def test_engine_initialization(self):
        """Test engine initializes with default config."""
        engine = EmbeddingEngine()
        
        assert engine.is_loaded is False
        assert engine.config.model_id == "Qwen/Qwen3-Embedding-8B"
    
    def test_engine_with_custom_config(self):
        """Test engine accepts custom configuration."""
        config = Settings(
            model_id="custom/model",
            device="cuda",
            default_dimension=1024,
        )
        engine = EmbeddingEngine(config)
        
        assert engine.config.model_id == "custom/model"
        assert engine.config.device == "cuda"
        assert engine.config.default_dimension == 1024
    
    def test_embed_single_text(self, engine_with_mock):
        """Test embedding a single text."""
        engine, mock_model = engine_with_mock
        mock_model.encode.return_value = np.random.randn(1, 4096).astype(np.float32)
        
        result = engine.embed("Hello world")
        
        assert isinstance(result, EmbeddingResult)
        assert result.num_texts == 1
        assert result.dimensions == 4096
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 4096
    
    def test_embed_multiple_texts(self, engine_with_mock):
        """Test embedding multiple texts."""
        engine, mock_model = engine_with_mock
        mock_model.encode.return_value = np.random.randn(3, 4096).astype(np.float32)
        
        result = engine.embed(["Text 1", "Text 2", "Text 3"])
        
        assert result.num_texts == 3
        assert len(result.embeddings) == 3
    
    def test_embed_with_instruction(self, engine_with_mock):
        """Test embedding with instruction formatting."""
        engine, mock_model = engine_with_mock
        mock_model.encode.return_value = np.random.randn(1, 4096).astype(np.float32)
        
        result = engine.embed(
            "What is ML?",
            instruction="Given a query, retrieve relevant documents"
        )
        
        # Check that encode was called with formatted text
        call_args = mock_model.encode.call_args[0][0]
        assert "Instruct:" in call_args[0]
        assert "Query:" in call_args[0]
    
    def test_embed_with_custom_dimension(self, engine_with_mock):
        """Test embedding with reduced dimensions (MRL)."""
        engine, mock_model = engine_with_mock
        mock_model.encode.return_value = np.random.randn(1, 4096).astype(np.float32)
        
        result = engine.embed("Test", dimension=1024)
        
        # Should truncate to requested dimension
        assert result.dimensions == 1024
        assert len(result.embeddings[0]) == 1024
    
    def test_get_model_info_not_loaded(self):
        """Test model info when model is not loaded."""
        engine = EmbeddingEngine()
        info = engine.get_model_info()
        
        assert info["status"] == "not_loaded"
        assert "model_id" in info
    
    def test_get_model_info_loaded(self, engine_with_mock):
        """Test model info when model is loaded."""
        engine, _ = engine_with_mock
        info = engine.get_model_info()
        
        assert info["status"] == "loaded"
        assert info["embedding_dimension"] == 4096


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""
    
    def test_result_structure(self):
        """Test EmbeddingResult has correct structure."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            dimensions=3,
            model_id="test",
            num_texts=1,
            normalized=True,
        )
        
        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.dimensions == 3
        assert result.model_id == "test"
        assert result.num_texts == 1
        assert result.normalized is True
