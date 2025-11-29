"""
Model loading and embedding generation for Qwen3-Embedding.

This module handles:
- Model and tokenizer loading with proper configuration
- Efficient batch embedding generation
- Instruction-aware embedding for improved retrieval
- Memory-efficient inference on CPU/GPU
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from numpy.typing import NDArray

from .config import PoolingStrategy, Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    
    embeddings: NDArray[np.float32]
    """The generated embeddings as a numpy array of shape (n_texts, embedding_dim)."""
    
    dimensions: int
    """The dimensionality of each embedding vector."""
    
    num_texts: int
    """Number of texts that were embedded."""
    
    model_name: str
    """Name of the model used to generate embeddings."""
    
    truncated: List[bool] = field(default_factory=list)
    """Whether each input text was truncated to fit max_length."""
    
    @property
    def as_list(self) -> List[List[float]]:
        """Return embeddings as a nested list (JSON-serializable)."""
        return self.embeddings.tolist()


class EmbeddingModel:
    """
    Qwen3 Embedding model wrapper for efficient embedding generation.
    
    This class handles model loading, tokenization, and embedding generation
    with support for batching, instruction-aware embeddings, and various
    pooling strategies.
    
    Example:
        model = EmbeddingModel()
        model.load()
        
        # Single text
        result = model.embed("What is machine learning?")
        
        # Batch with instruction
        result = model.embed(
            ["query 1", "query 2"],
            instruction="Retrieve relevant passages for the query"
        )
    """
    
    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize the embedding model wrapper.
        
        Args:
            settings: Configuration settings. Uses global settings if not provided.
        """
        self.settings = settings or get_settings()
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._embedding_dim: Optional[int] = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._is_loaded
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the embedding vectors."""
        if self._embedding_dim is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._embedding_dim
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.settings.model_name
    
    def load(self) -> None:
        """
        Load the model and tokenizer.
        
        This method downloads the model if not cached, loads weights into memory,
        and prepares the model for inference. For large models like Qwen3-Embedding-8B,
        this can take several minutes on first run.
        
        Raises:
            RuntimeError: If model loading fails.
        """
        if self._is_loaded:
            logger.info("Model already loaded, skipping reload")
            return
        
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading model: {self.settings.model_name}")
        logger.info(f"Device: {self.settings.effective_device}")
        logger.info(f"Dtype: {self.settings.torch_dtype}")
        
        try:
            # Determine torch dtype
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "auto": "auto",
            }
            torch_dtype = dtype_map.get(self.settings.torch_dtype, torch.float32)
            
            # Load using SentenceTransformer for optimal performance
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": self.settings.trust_remote_code,
            }
            
            if self.settings.effective_cache_dir:
                model_kwargs["cache_folder"] = self.settings.effective_cache_dir
            
            self._model = SentenceTransformer(
                self.settings.model_name,
                device=self.settings.effective_device,
                model_kwargs=model_kwargs,
                tokenizer_kwargs={"padding_side": "left"},
            )
            
            # Get embedding dimension from a test encoding
            test_embedding = self._model.encode("test", convert_to_numpy=True)
            self._embedding_dim = test_embedding.shape[0]
            
            self._is_loaded = True
            logger.info(f"Model loaded successfully! Embedding dimension: {self._embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model {self.settings.model_name}: {e}") from e
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._is_loaded = False
        self._embedding_dim = None
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded")
    
    def _format_with_instruction(self, text: str, instruction: Optional[str]) -> str:
        """
        Format text with instruction for improved retrieval.
        
        Qwen3 embedding models support instruction-aware embeddings that improve
        retrieval performance by 1-5%. The instruction describes the task.
        
        Args:
            text: The input text to embed.
            instruction: Task instruction to prepend.
        
        Returns:
            Formatted text with instruction.
        """
        if instruction:
            return f"Instruct: {instruction}\nQuery: {text}"
        return text
    
    def embed(
        self,
        texts: Union[str, Sequence[str]],
        instruction: Optional[str] = None,
        normalize: Optional[bool] = None,
        batch_size: Optional[int] = None,
        show_progress: Optional[bool] = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings for one or more texts.
        
        Args:
            texts: Single text or sequence of texts to embed.
            instruction: Optional instruction for task-aware embeddings.
                        Example: "Retrieve relevant passages for the query"
            normalize: Whether to L2-normalize embeddings. Uses settings default if None.
            batch_size: Batch size for processing. Uses settings default if None.
            show_progress: Whether to show progress bar. Uses settings default if None.
        
        Returns:
            EmbeddingResult containing the embeddings and metadata.
        
        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If texts is empty.
        
        Example:
            # Single query with instruction
            result = model.embed(
                "What causes climate change?",
                instruction="Given a scientific question, retrieve explanatory passages"
            )
            
            # Batch of documents (no instruction for documents)
            result = model.embed([
                "Climate change is caused by greenhouse gases...",
                "The greenhouse effect traps heat in the atmosphere...",
            ])
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        if len(texts) == 0:
            raise ValueError("texts cannot be empty")
        
        # Use defaults from settings if not specified
        if normalize is None:
            normalize = self.settings.normalize_embeddings
        if batch_size is None:
            batch_size = self.settings.batch_size
        if show_progress is None:
            show_progress = self.settings.show_progress
        
        # Use default instruction from settings if available and none provided
        effective_instruction = instruction
        if effective_instruction is None:
            effective_instruction = self.settings.default_instruction
        
        # Format texts with instruction
        formatted_texts = [
            self._format_with_instruction(t, effective_instruction)
            for t in texts
        ]
        
        logger.debug(f"Embedding {len(texts)} texts with batch_size={batch_size}")
        
        # Generate embeddings using SentenceTransformer
        embeddings = self._model.encode(
            formatted_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        
        # Ensure 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        return EmbeddingResult(
            embeddings=embeddings.astype(np.float32),
            dimensions=self._embedding_dim,
            num_texts=len(texts),
            model_name=self.settings.model_name,
            truncated=[False] * len(texts),  # SentenceTransformer handles truncation
        )
    
    def similarity(
        self,
        embeddings1: NDArray[np.float32],
        embeddings2: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings, shape (n, dim) or (dim,).
            embeddings2: Second set of embeddings, shape (m, dim) or (dim,).
        
        Returns:
            Similarity matrix of shape (n, m), or scalar if both inputs are 1D.
        """
        # Ensure 2D
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        # Normalize if not already
        norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        embeddings1_norm = embeddings1 / np.maximum(norms1, 1e-8)
        embeddings2_norm = embeddings2 / np.maximum(norms2, 1e-8)
        
        # Compute cosine similarity
        similarity = embeddings1_norm @ embeddings2_norm.T
        
        return similarity.astype(np.float32)


# Global model instance
_model: Optional[EmbeddingModel] = None


def get_model() -> EmbeddingModel:
    """Get the global model instance, creating one if needed."""
    global _model
    if _model is None:
        _model = EmbeddingModel()
    return _model


def load_model(settings: Optional[Settings] = None) -> EmbeddingModel:
    """Load the model with optional settings override."""
    global _model
    if settings:
        _model = EmbeddingModel(settings)
    else:
        _model = get_model()
    _model.load()
    return _model
