"""
Domain-Tuned Reranking Engine for Information Lensing.

Simple interface: score_pair(code_a, code_b) → probability (0.0-1.0)

Uses official Qwen3-Reranker format with custom instruction for domain-aware scoring.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    """Result of scoring a code pair."""
    score: float
    query: str
    document: str


@dataclass
class BatchScoreResult:
    """Result of scoring multiple pairs."""
    scores: list[float]
    pairs: list[tuple[str, str]]
    symmetric: bool


class RerankerEngine:
    """
    Domain-tuned reranker for Information Lensing.
    
    Uses official Qwen3-Reranker format with custom instruction
    to detect semantic domain similarity vs structural similarity.
    """
    
    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or get_settings()
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        
        # Qwen3-Reranker specific
        self._prefix_tokens = None
        self._suffix_tokens = None
        self._token_true_id = None
        self._token_false_id = None
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    @property
    def model(self):
        if not self._is_loaded:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        if not self._is_loaded:
            self._load_model()
        return self._tokenizer
    
    def _get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.torch_dtype, torch.float32)
    
    def _get_device(self) -> str:
        device = self.config.device
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
    
    def _load_model(self) -> None:
        """Load Qwen3-Reranker with official format."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model: {self.config.model_id}")
        logger.info(f"Device: {self._get_device()}, Dtype: {self.config.torch_dtype}")
        
        try:
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="left",
                cache_dir=self.config.cache_dir,
            )
            
            # Load model
            logger.info("Loading model weights...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                dtype=self._get_torch_dtype(),
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
            )
            
            device = self._get_device()
            self._model = self._model.to(device)
            self._model.eval()
            
            for param in self._model.parameters():
                param.requires_grad = False
            
            # Setup official Qwen3-Reranker tokens
            self._setup_tokens()
            
            self._is_loaded = True
            
            num_params = sum(p.numel() for p in self._model.parameters())
            logger.info(f"Model loaded: {num_params:,} parameters on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def _setup_tokens(self) -> None:
        """Setup official Qwen3-Reranker prefix/suffix and yes/no token IDs."""
        # Official prefix from HuggingFace
        prefix = (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".'
            '<|im_end|>\n'
            '<|im_start|>user\n'
        )
        
        # Official suffix
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        
        self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)
        
        # Token IDs for yes/no
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        
        logger.info(f"Token IDs: yes={self._token_true_id}, no={self._token_false_id}")
    
    def _format_instruction(self, query: str, document: str) -> str:
        """Format using official Qwen3-Reranker format with custom instruction."""
        if self.config.use_custom_instruction:
            instruction = self.config.custom_instruction
        else:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    
    def _process_input(self, query: str, document: str) -> dict:
        """Process input with prefix/suffix tokens (official method)."""
        formatted = self._format_instruction(query, document)
        
        # Tokenize
        max_content_len = self.config.max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        inputs = self._tokenizer(
            formatted,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=max_content_len,
        )
        
        # Add prefix and suffix tokens
        input_ids = self._prefix_tokens + inputs['input_ids'] + self._suffix_tokens
        
        device = self._get_device()
        return {
            'input_ids': torch.tensor([input_ids], dtype=torch.long, device=device),
            'attention_mask': torch.ones(1, len(input_ids), dtype=torch.long, device=device),
        }
    
    @torch.no_grad()
    def _compute_score(self, inputs: dict) -> float:
        """
        Compute score using OFFICIAL Qwen3-Reranker method.
        
        From HuggingFace:
            batch_scores = model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        """
        outputs = self._model(**inputs)
        logits = outputs.logits[:, -1, :]
        
        # Get logits for yes/no tokens
        true_logit = logits[:, self._token_true_id]
        false_logit = logits[:, self._token_false_id]
        
        # Stack [false, true] and compute probability
        stacked = torch.stack([false_logit, true_logit], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        
        # Return P(yes) = exp(log_prob[1])
        score = log_probs[:, 1].exp().item()
        
        return float(score)
    
    def score_pair(self, query: str, document: str) -> ScoreResult:
        """
        Score semantic similarity between two code segments.
        
        Args:
            query: First code segment
            document: Second code segment
        
        Returns:
            ScoreResult with probability (0.0-1.0)
        """
        logger.debug(f"Scoring pair: {len(query)} × {len(document)} chars")
        
        inputs = self._process_input(query, document)
        score = self._compute_score(inputs)
        
        return ScoreResult(score=score, query=query, document=document)
    
    def score_pair_raw(self, query: str, document: str) -> float:
        """Just return the probability score."""
        return self.score_pair(query, document).score
    
    def score_pair_symmetric(self, code_a: str, code_b: str) -> float:
        """
        Score with symmetrization: (score(a,b) + score(b,a)) / 2
        
        Rerankers are inherently asymmetric (query vs document).
        For Information Lensing we need symmetric similarity matrix,
        so we average both directions.
        
        Args:
            code_a: First code segment
            code_b: Second code segment
        
        Returns:
            Symmetric probability score (0.0-1.0)
        """
        score_ab = self.score_pair_raw(code_a, code_b)
        score_ba = self.score_pair_raw(code_b, code_a)
        return (score_ab + score_ba) / 2
    
    def score_batch(
        self,
        pairs: list[tuple[str, str]],
        symmetric: bool = True,
    ) -> BatchScoreResult:
        """
        Score multiple pairs efficiently.
        
        Args:
            pairs: List of (code_a, code_b) tuples
            symmetric: If True, compute (score(a,b) + score(b,a)) / 2
        
        Returns:
            BatchScoreResult with scores in same order as input pairs
        """
        logger.info(f"Scoring batch of {len(pairs)} pairs (symmetric={symmetric})")
        
        scores = []
        for i, (code_a, code_b) in enumerate(pairs):
            if symmetric:
                score = self.score_pair_symmetric(code_a, code_b)
            else:
                score = self.score_pair_raw(code_a, code_b)
            scores.append(score)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i + 1}/{len(pairs)} pairs")
        
        logger.info(f"Batch complete: {len(scores)} scores computed")
        
        return BatchScoreResult(
            scores=scores,
            pairs=pairs,
            symmetric=symmetric,
        )
    
    def get_model_info(self) -> dict:
        """Get model status."""
        if not self._is_loaded:
            return {"status": "not_loaded", "model_id": self.config.model_id}
        
        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype
        num_params = sum(p.numel() for p in self._model.parameters())
        
        return {
            "status": "loaded",
            "model_id": self.config.model_id,
            "device": str(device),
            "dtype": str(dtype),
            "num_parameters": f"{num_params:,}",
            "max_length": self.config.max_length,
            "custom_instruction_enabled": self.config.use_custom_instruction,
            "token_true_id": self._token_true_id,
            "token_false_id": self._token_false_id,
        }
    
    def warmup(self) -> None:
        """Warm up model."""
        logger.info("Warming up...")
        _ = self.score_pair("test query", "test document")
        logger.info("Warmup complete")
    
    def unload(self) -> None:
        """Unload model."""
        if self._is_loaded:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._is_loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")


_engine: RerankerEngine | None = None


def get_engine() -> RerankerEngine:
    global _engine
    if _engine is None:
        _engine = RerankerEngine()
    return _engine


def reset_engine() -> None:
    global _engine
    if _engine is not None:
        _engine.unload()
    _engine = None
