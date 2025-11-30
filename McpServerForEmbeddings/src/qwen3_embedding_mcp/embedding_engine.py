"""
Embedding Engine for Qwen3-Embedding-8B.

Handles model loading and embedding generation.
Designed for CPU inference with large RAM (optimized for systems with 64GB+ RAM).

Supports Information Lensing via instruction-aware embeddings:
- structural: graph topology, architectural position
- semantic: business domain, code meaning
- behavioral: runtime patterns, execution flow
"""

import logging
from typing import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import Settings, settings, DOMAIN_INSTRUCTIONS, LensType

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    embeddings: list[list[float]]
    dimensions: int
    model_id: str
    num_texts: int
    normalized: bool
    lens: str | None = None  # Which lens was applied


class EmbeddingEngine:
    """
    High-performance embedding engine using Qwen3-Embedding-8B.
    
    Features:
    - Lazy model loading (loads on first use)
    - Configurable embedding dimensions via MRL (Matryoshka Representation Learning)
    - Information Lensing via instruction-aware embeddings
    - Batch processing with automatic chunking
    - Thread-safe for concurrent requests
    
    Information Lensing:
        The engine supports three "gravitational lenses" that curve the embedding
        space toward different aspects of code semantics:
        
        - structural: Graph topology, architectural position, connectivity
        - semantic: Business domain, intent, conceptual meaning  
        - behavioral: Runtime patterns, execution flow, side effects
        
        Reference: Marchewka (2025), "Information Lensing: A Gravitational 
        Approach to Domain-Specific Embedding Transformation"
    
    Example:
        >>> engine = EmbeddingEngine()
        >>> result = engine.embed("class PaymentService...", lens="semantic")
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
    
    def _format_instruction(self, instruction: str, text: str) -> str:
        """Format text with instruction prefix (Qwen3 format)."""
        return f"Instruct: {instruction}\nQuery: {text}"
    
    def _get_lens_instruction(self, lens: LensType) -> str:
        """Get the domain instruction for a specific lens type."""
        if lens not in DOMAIN_INSTRUCTIONS:
            raise ValueError(f"Unknown lens type: {lens}. Valid: {list(DOMAIN_INSTRUCTIONS.keys())}")
        return DOMAIN_INSTRUCTIONS[lens]
    
    def embed(
        self,
        texts: str | Sequence[str],
        *,
        lens: LensType | None = None,
        instruction: str | None = None,
        dimension: int | None = None,
        normalize: bool | None = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings for input text(s).
        
        Information Lensing Mode (recommended for code):
            Use the `lens` parameter to apply domain-specific embedding focus:
            - "structural": Graph topology, architectural position
            - "semantic": Business domain, code meaning
            - "behavioral": Runtime patterns, execution flow
        
        Args:
            texts: Single text or list of texts to embed (max 32K tokens each).
            lens: Information Lensing type ("structural", "semantic", "behavioral").
                  If provided, applies domain-specific instruction automatically.
                  Overrides instruction.
            instruction: Custom instruction to prepend (for advanced use).
            dimension: Output embedding dimension (uses MRL truncation if < max).
            normalize: Whether to L2-normalize embeddings.
            
        Returns:
            EmbeddingResult containing embeddings and metadata.
            
        Example (Information Lensing):
            >>> # Embed graph structure data
            >>> result = engine.embed(graph_context, lens="structural")
            
            >>> # Embed source code
            >>> result = engine.embed(code_content, lens="semantic")
            
            >>> # Embed runtime behavior data
            >>> result = engine.embed(behavior_data, lens="behavioral")
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        texts = list(texts)
        logger.debug(f"Processing {len(texts)} texts")
        
        # Determine instruction source (priority: lens > instruction)
        effective_lens: str | None = None
        
        if lens:
            # Information Lensing mode - use domain instruction
            lens_instruction = self._get_lens_instruction(lens)
            texts = [self._format_instruction(lens_instruction, text) for text in texts]
            effective_lens = lens
            logger.debug(f"Applied {lens} lens instruction")
        elif instruction:
            # Custom instruction mode
            texts = [self._format_instruction(instruction, text) for text in texts]
        
        # Get settings
        dim = dimension or self.config.default_dimension
        norm = normalize if normalize is not None else self.config.normalize_embeddings
        
        # Generate embeddings
        encode_kwargs = {
            "normalize_embeddings": norm,
            "batch_size": self.config.batch_size,
            "show_progress_bar": self.config.show_progress,
        }
        
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
            lens=effective_lens,
        )
    
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
                "available_lenses": list(DOMAIN_INSTRUCTIONS.keys()),
            }
        
        return {
            "status": "loaded",
            "model_id": self.config.model_id,
            "device": str(self.model.device),
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "dtype": str(self._get_torch_dtype()),
            "available_lenses": list(DOMAIN_INSTRUCTIONS.keys()),
        }
    
    def warmup(self) -> None:
        """
        Warm up the model by running a test inference.
        
        Call this after loading to ensure the model is fully initialized
        and to get more consistent latency measurements.
        """
        logger.info("Warming up model...")
        _ = self.embed("Warmup text for model initialization.", lens="semantic")
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
