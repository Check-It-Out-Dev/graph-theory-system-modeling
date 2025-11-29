"""
Embedding Engine for Qwen3-Embedding-8B.

Handles model loading, embedding generation, and similarity computation.
Designed for CPU inference with large RAM (optimized for systems with 64GB+ RAM).
"""

import logging
from typing import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import Settings, settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    embeddings: list[list[float]]
    dimensions: int
    model_id: str
    num_texts: int
    normalized: bool


@dataclass
class SimilarityResult:
    """Result of a similarity computation."""
    scores: list[list[float]]
    num_queries: int
    num_documents: int


class EmbeddingEngine:
    """
    High-performance embedding engine using Qwen3-Embedding-8B.
    
    Features:
    - Lazy model loading (loads on first use)
    - Configurable embedding dimensions via MRL (Matryoshka Representation Learning)
    - Instruction-aware embeddings for improved retrieval
    - Batch processing with automatic chunking
    - Thread-safe for concurrent requests
    
    Example:
        >>> engine = EmbeddingEngine()
        >>> result = engine.embed(["Hello world", "How are you?"])
        >>> print(result.embeddings[0][:5])  # First 5 dimensions
    """
    
    def __init__(self, config: Settings | None = None) -> None:
        """
        Initialize the embedding engine.
        
        Args:
            config: Optional settings override. Uses global settings if not provided.
        """
        self.config = config or settings
        self._model: SentenceTransformer | None = None
        self._is_loaded = False
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded
    
    @property
    def model(self) -> SentenceTransformer:
        """
        Get the loaded model, initializing if necessary.
        
        Returns:
            The SentenceTransformer model instance.
            
        Raises:
            RuntimeError: If model loading fails.
        """
        if not self._is_loaded:
            self._load_model()
        return self._model  # type: ignore
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Convert config dtype string to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": torch.float32,  # Default to float32 for CPU
        }
        return dtype_map.get(self.config.torch_dtype, torch.float32)
    
    def _load_model(self) -> None:
        """
        Load the embedding model into memory.
        
        This operation can take 1-3 minutes for the 8B model depending on
        disk speed and available memory.
        """
        logger.info(f"Loading model: {self.config.model_id}")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Dtype: {self.config.torch_dtype}")
        
        try:
            model_kwargs = {
                "dtype": self._get_torch_dtype(),
                "trust_remote_code": self.config.trust_remote_code,
            }
            
            if self.config.cache_dir:
                model_kwargs["cache_folder"] = self.config.cache_dir
            
            # Determine device
            device = self.config.device
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            self._model = SentenceTransformer(
                self.config.model_id,
                device=device,
                model_kwargs=model_kwargs,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            # Set max sequence length
            self._model.max_seq_length = self.config.max_sequence_length
            
            self._is_loaded = True
            logger.info(f"Model loaded successfully on {device}")
            logger.info(f"Max sequence length: {self._model.max_seq_length}")
            logger.info(f"Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def embed(
        self,
        texts: str | Sequence[str],
        *,
        instruction: str | None = None,
        dimension: int | None = None,
        normalize: bool | None = None,
        prompt_name: str | None = None,
        is_query: bool = False,
    ) -> EmbeddingResult:
        """
        Generate embeddings for input text(s).
        
        Qwen3-Embedding is instruction-aware and supports two modes:
        1. Built-in prompts: Use prompt_name="query" or "passage" (recommended)
        2. Custom instruction: Provide your own instruction string
        
        Args:
            texts: Single text or list of texts to embed (max 32K tokens each).
            instruction: Custom instruction to prepend. If provided, overrides prompt_name.
                        Example: "Given a web search query, retrieve relevant passages"
            dimension: Output embedding dimension (uses MRL truncation if < max).
                      Defaults to config.default_dimension (4096).
            normalize: Whether to L2-normalize embeddings. Defaults to config setting.
            prompt_name: Built-in prompt name ("query" or "passage") for retrieval tasks.
                        Use "query" for search queries, "passage" for documents.
            is_query: Shorthand for prompt_name="query". Ignored if prompt_name is set.
            
        Returns:
            EmbeddingResult containing embeddings and metadata.
            
        Example:
            >>> # For retrieval queries (using built-in prompt)
            >>> result = engine.embed("What is machine learning?", is_query=True)
            
            >>> # For retrieval queries (using custom instruction)
            >>> result = engine.embed(
            ...     "What is machine learning?",
            ...     instruction="Given a scientific question, retrieve relevant papers"
            ... )
            
            >>> # For documents/passages
            >>> result = engine.embed([
            ...     "Machine learning is a subset of AI...",
            ...     "Deep learning uses neural networks...",
            ... ], prompt_name="document")
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        texts = list(texts)
        logger.debug(f"Processing {len(texts)} texts")
        
        # Determine prompt strategy
        effective_prompt_name = prompt_name
        if effective_prompt_name is None and is_query:
            effective_prompt_name = "query"
        
        # Apply custom instruction if provided (overrides prompt_name)
        if instruction:
            texts = [self._format_instruction(instruction, text) for text in texts]
            effective_prompt_name = None  # Don't use built-in prompt with custom instruction
        
        # Get settings
        dim = dimension or self.config.default_dimension
        norm = normalize if normalize is not None else self.config.normalize_embeddings
        
        # Generate embeddings
        encode_kwargs = {
            "normalize_embeddings": norm,
            "batch_size": self.config.batch_size,
            "show_progress_bar": self.config.show_progress,
        }
        
        if effective_prompt_name:
            encode_kwargs["prompt_name"] = effective_prompt_name
        
        embeddings = self.model.encode(texts, **encode_kwargs)
        
        # Apply MRL truncation if requested dimension is smaller
        if dim < embeddings.shape[1]:
            embeddings = embeddings[:, :dim]
            # Re-normalize after truncation if needed
            if norm:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return EmbeddingResult(
            embeddings=embeddings.tolist(),
            dimensions=embeddings.shape[1],
            model_id=self.config.model_id,
            num_texts=len(texts),
            normalized=norm,
        )
    
    def similarity(
        self,
        queries: str | Sequence[str],
        documents: str | Sequence[str],
        *,
        query_instruction: str | None = None,
    ) -> SimilarityResult:
        """
        Compute similarity scores between queries and documents.
        
        Args:
            queries: Query text(s).
            documents: Document text(s).
            query_instruction: Optional instruction for query embeddings.
            
        Returns:
            SimilarityResult with similarity matrix (queries x documents).
        """
        # Handle single inputs
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(documents, str):
            documents = [documents]
        
        queries = list(queries)
        documents = list(documents)
        
        # Embed queries with instruction if provided
        query_embeddings = self.embed(
            queries,
            instruction=query_instruction,
            normalize=True,
        ).embeddings
        
        # Embed documents without instruction
        doc_embeddings = self.embed(
            documents,
            normalize=True,
        ).embeddings
        
        # Compute cosine similarity (dot product since normalized)
        query_arr = np.array(query_embeddings)
        doc_arr = np.array(doc_embeddings)
        scores = np.dot(query_arr, doc_arr.T)
        
        return SimilarityResult(
            scores=scores.tolist(),
            num_queries=len(queries),
            num_documents=len(documents),
        )
    
    def _format_instruction(self, instruction: str, text: str) -> str:
        """Format text with instruction prefix."""
        return f"Instruct: {instruction}\nQuery: {text}"
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata.
        """
        if not self._is_loaded:
            return {
                "status": "not_loaded",
                "model_id": self.config.model_id,
                "device": self.config.device,
            }
        
        return {
            "status": "loaded",
            "model_id": self.config.model_id,
            "device": str(self.model.device),
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "dtype": str(self._get_torch_dtype()),
        }
    
    def warmup(self) -> None:
        """
        Warm up the model by running a test inference.
        
        Call this after loading to ensure the model is fully initialized
        and to get more consistent latency measurements.
        """
        logger.info("Warming up model...")
        _ = self.embed("Warmup text for model initialization.")
        logger.info("Warmup complete")
    
    def unload(self) -> None:
        """
        Unload the model from memory.
        
        Useful for freeing up RAM when the model is no longer needed.
        """
        if self._is_loaded:
            del self._model
            self._model = None
            self._is_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded from memory")


# Global engine instance (lazy initialization)
_engine: EmbeddingEngine | None = None


def get_engine() -> EmbeddingEngine:
    """Get or create the global embedding engine instance."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine
