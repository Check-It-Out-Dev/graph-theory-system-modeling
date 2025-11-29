"""
Tests for the Qwen3-Reranker MCP Server.

These tests verify the configuration, engine initialization, and basic functionality.
Integration tests that load the actual model are marked with @pytest.mark.slow.
"""

import pytest
from unittest.mock import MagicMock, patch

from qwen3_reranker_mcp.config import Settings, get_settings, reload_settings
from qwen3_reranker_mcp.reranker_engine import (
    RerankerEngine,
    RerankResult,
    SingleScoreResult,
    BatchScoreResult,
)


class TestSettings:
    """Tests for configuration settings."""
    
    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        
        assert settings.model_id == "Qwen/Qwen3-Reranker-8B"
        assert settings.device == "cpu"
        assert settings.torch_dtype == "float32"
        assert settings.max_length == 8192
        assert settings.batch_size == 8
        assert settings.default_top_k == 10
        assert settings.log_level == "INFO"
    
    def test_settings_from_env(self, monkeypatch):
        """Test settings can be loaded from environment variables."""
        monkeypatch.setenv("QWEN3_RERANKER_DEVICE", "cuda")
        monkeypatch.setenv("QWEN3_RERANKER_BATCH_SIZE", "16")
        monkeypatch.setenv("QWEN3_RERANKER_LOG_LEVEL", "DEBUG")
        
        settings = Settings()
        
        assert settings.device == "cuda"
        assert settings.batch_size == 16
        assert settings.log_level == "DEBUG"
    
    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        reload_settings()  # Reset to ensure clean state
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2


class TestRerankerEngine:
    """Tests for the reranker engine."""
    
    def test_engine_initialization(self):
        """Test engine can be created without loading model."""
        engine = RerankerEngine()
        
        assert not engine.is_loaded
        assert engine.config is not None
    
    def test_engine_with_custom_config(self):
        """Test engine with custom settings."""
        config = Settings(batch_size=4, max_length=4096)
        engine = RerankerEngine(config)
        
        assert engine.config.batch_size == 4
        assert engine.config.max_length == 4096


class TestResultDataClasses:
    """Tests for result data classes."""
    
    def test_single_score_result(self):
        """Test SingleScoreResult structure."""
        result = SingleScoreResult(
            score=0.85,
            query="What is AI?",
            document="Artificial intelligence is...",
        )
        
        assert result.score == 0.85
        assert result.query == "What is AI?"
        assert "score" in result.as_dict
    
    def test_batch_score_result(self):
        """Test BatchScoreResult structure."""
        result = BatchScoreResult(
            scores=[0.8, 0.3],
            pairs=[("q1", "d1"), ("q2", "d2")],
            num_pairs=2,
        )
        
        assert len(result.scores) == 2
        assert result.num_pairs == 2
    
    def test_rerank_result(self):
        """Test RerankResult structure."""
        result = RerankResult(
            scores=[0.9, 0.7, 0.5],
            indices=[2, 0, 1],
            documents=["doc2", "doc0", "doc1"],
            query="test query",
            num_documents=3,
            top_k=3,
        )
        
        assert len(result.scores) == 3
        assert result.scores[0] == 0.9
        assert result.indices[0] == 2
        assert result.query == "test query"


class TestEngineWithMockedModel:
    """Tests using mocked model to avoid loading the real one."""
    
    @pytest.fixture
    def mocked_engine(self):
        """Create engine with mocked model."""
        engine = RerankerEngine()
        
        # Mock the model loading
        engine._is_loaded = True
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        
        return engine
    
    def test_get_model_info_loaded(self, mocked_engine):
        """Test model info when loaded."""
        # Setup mock
        mocked_engine._model.parameters.return_value = [
            MagicMock(numel=lambda: 1000, device="cpu", dtype="float32")
        ]
        
        info = mocked_engine.get_model_info()
        
        assert info["status"] == "loaded"
    
    def test_get_model_info_not_loaded(self):
        """Test model info when not loaded."""
        engine = RerankerEngine()
        info = engine.get_model_info()
        
        assert info["status"] == "not_loaded"
        assert "model_id" in info


class TestInputValidation:
    """Tests for input validation."""
    
    def test_empty_documents_raises(self):
        """Test that empty documents list raises error."""
        engine = RerankerEngine()
        engine._is_loaded = True
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        
        with pytest.raises(ValueError, match="documents cannot be empty"):
            engine.rerank(query="test", documents=[])
    
    def test_empty_pairs_returns_empty(self):
        """Test that empty pairs list returns empty result."""
        engine = RerankerEngine()
        engine._is_loaded = True
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        
        result = engine.score_pairs([])
        assert result.num_pairs == 0
        assert result.scores == []


@pytest.mark.slow
@pytest.mark.integration
class TestIntegration:
    """
    Integration tests that load the actual model.
    
    These tests are slow and require significant RAM.
    Skip with: pytest -m "not slow"
    """
    
    @pytest.fixture(scope="class")
    def engine(self):
        """Create and load engine for integration tests."""
        engine = RerankerEngine()
        # Model will be loaded lazily on first use
        yield engine
        engine.unload()
    
    def test_score_pair_basic(self, engine):
        """Test basic single pair scoring."""
        result = engine.score_pair(
            query="What is machine learning?",
            document="Machine learning is a subset of AI that enables systems to learn from data.",
        )
        
        assert 0 <= result.score <= 1
        assert result.query == "What is machine learning?"
    
    def test_score_pair_semantic_difference(self, engine):
        """Test that semantically different content scores lower."""
        result_relevant = engine.score_pair(
            query="What is machine learning?",
            document="Machine learning is a subset of artificial intelligence.",
        )
        
        result_irrelevant = engine.score_pair(
            query="What is machine learning?",
            document="Pizza is a delicious Italian food with cheese and tomato sauce.",
        )
        
        # Relevant should score higher than irrelevant
        assert result_relevant.score > result_irrelevant.score
    
    def test_score_pairs_batch(self, engine):
        """Test batch pair scoring."""
        result = engine.score_pairs([
            ("What is AI?", "Artificial intelligence is the simulation of human intelligence."),
            ("What is AI?", "Pizza is a delicious Italian food."),
        ])
        
        assert len(result.scores) == 2
        # First pair should score higher (more relevant)
        assert result.scores[0] > result.scores[1]
    
    def test_rerank_basic(self, engine):
        """Test basic reranking functionality."""
        result = engine.rerank(
            query="What is machine learning?",
            documents=[
                "Machine learning is a subset of AI that enables systems to learn.",
                "The weather today is sunny and warm.",
                "Deep learning uses neural networks with many layers.",
            ],
            top_k=2,
        )
        
        assert len(result.scores) == 2
        assert all(0 <= s <= 1 for s in result.scores)
        assert result.scores[0] >= result.scores[1]  # Sorted descending
    
    def test_information_lensing_workflow(self, engine):
        """
        Test the Information Lensing workflow:
        Two similar-looking code segments that should have low reranker score.
        """
        # These would have HIGH embedding similarity due to similar syntax
        code_a = """
        public class PaymentService {
            public void processPayment(BigDecimal amount, String currency) {
                // Process payment logic
                validateAmount(amount);
                chargeCard(amount, currency);
            }
        }
        """
        
        code_b = """
        public class InventoryService {
            public void updateStock(int quantity, String location) {
                // Update inventory logic
                validateQuantity(quantity);
                adjustStock(quantity, location);
            }
        }
        """
        
        result = engine.score_pair(query=code_a, document=code_b)
        
        # These are semantically DIFFERENT (different domains)
        # Reranker should give them a LOW score despite similar syntax
        # In Information Lensing, the divergence = embedding_sim - reranker_score would be high
        assert result.score < 0.7  # Should not be highly similar
        
        # For comparison, similar domain code should score higher
        code_c = """
        public class RefundService {
            public void processRefund(BigDecimal amount, String currency) {
                // Process refund logic  
                validateAmount(amount);
                refundToCard(amount, currency);
            }
        }
        """
        
        result_similar = engine.score_pair(query=code_a, document=code_c)
        
        # Payment and Refund are semantically similar (same domain)
        assert result_similar.score > result.score
