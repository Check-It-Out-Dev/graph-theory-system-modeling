"""
OpenAI-Compatible REST API for Qwen3-Embedding with Information Lensing.

Enables direct Neo4j APOC integration without LLM context pollution.
Endpoint: POST /v1/embeddings

Usage with Neo4j APOC:
    CALL apoc.ml.openai.embedding(['text'], 'x', {endpoint: 'http://localhost:7999/v1'})
    
Reference: Marchewka (2025), "Information Lensing"

Run:
    python -m qwen3_embedding_mcp --rest
    python -m qwen3_embedding_mcp --rest --port 7999 --host 0.0.0.0
"""

import logging
from typing import Union
from datetime import datetime

from flask import Flask, request, jsonify, Response
from pydantic import BaseModel, field_validator

from .embedding_engine import get_engine, EmbeddingEngine
from .config import (
    settings, 
    DOMAIN_INSTRUCTIONS, 
    LensType,
    EMBEDDING_SERVER_VERSION,
    MODEL_VERSION,
    LENS_HASHES,
    get_embedding_version_metadata,
)

logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__)


# =============================================================================
# Pydantic Models for Request/Response Validation
# =============================================================================

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    input: Union[str, list[str]]
    model: str = "semantic"  # Lens type (semantic, structural, behavioral)
    dimensions: int = 4096
    encoding_format: str = "float"
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v: str) -> str:
        valid_lenses = list(DOMAIN_INSTRUCTIONS.keys())
        if v not in valid_lenses:
            raise ValueError(f"Invalid model/lens: {v}. Valid options: {valid_lenses}")
        return v
    
    @field_validator('dimensions')
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        if not 128 <= v <= 4096:
            raise ValueError("dimensions must be between 128 and 4096")
        return v


# =============================================================================
# Global Engine Reference (Singleton)
# =============================================================================

_engine: EmbeddingEngine | None = None


def get_or_load_engine() -> EmbeddingEngine:
    """Get engine, loading model if needed (singleton pattern)."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    if not _engine.is_loaded:
        logger.info("Loading embedding model (first request)...")
        _ = _engine.model  # Triggers lazy load
        _engine.warmup()
    return _engine


# =============================================================================
# REST API Endpoints
# =============================================================================

@app.route('/v1/embeddings', methods=['POST'])
def create_embeddings() -> tuple[Response, int]:
    """
    OpenAI-compatible embeddings endpoint.
    
    Maps 'model' parameter to Information Lens type:
    - semantic: Business logic, domain concepts
    - structural: Graph topology, architecture
    - behavioral: Runtime patterns, execution flow
    
    Request:
        {
            "input": "text" or ["text1", "text2", ...],
            "model": "semantic" | "structural" | "behavioral",
            "dimensions": 4096  (optional, 128-4096)
        }
        
    Response:
        {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [...], "index": 0}],
            "model": "semantic",
            "usage": {"prompt_tokens": N, "total_tokens": N}
        }
    """
    try:
        # Parse and validate request
        data = request.get_json()
        if not data:
            return jsonify({"error": {"message": "Request body required", "type": "invalid_request_error"}}), 400
        
        try:
            req = EmbeddingRequest(**data)
        except ValueError as e:
            return jsonify({"error": {"message": str(e), "type": "invalid_request_error"}}), 422
        
        # Normalize input to list
        texts = req.input if isinstance(req.input, list) else [req.input]
        
        if not texts or all(not t.strip() for t in texts):
            return jsonify({"error": {"message": "Input cannot be empty", "type": "invalid_request_error"}}), 400
        
        # Get engine and generate embeddings
        engine = get_or_load_engine()
        
        logger.info(f"Generating {len(texts)} embedding(s) with '{req.model}' lens, dim={req.dimensions}")
        
        result = engine.embed(
            texts=texts,
            lens=req.model,  # model param = lens type
            dimension=req.dimensions,
        )
        
        # Build OpenAI-compatible response
        response_data = []
        for i, emb in enumerate(result.embeddings):
            response_data.append({
                "object": "embedding",
                "embedding": emb,
                "index": i
            })
        
        # Estimate token count (rough: ~4 chars per token for code)
        total_chars = sum(len(t) for t in texts)
        estimated_tokens = max(1, total_chars // 4)
        
        response = {
            "object": "list",
            "data": response_data,
            "model": req.model,
            "usage": {
                "prompt_tokens": estimated_tokens,
                "total_tokens": estimated_tokens
            }
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": {"message": str(e), "type": "invalid_request_error"}}), 422
    except Exception as e:
        logger.exception(f"Embedding generation failed: {e}")
        return jsonify({"error": {"message": f"Internal error: {str(e)}", "type": "server_error"}}), 500


@app.route('/v1/models', methods=['GET'])
def list_models() -> tuple[Response, int]:
    """
    List available models (lenses).
    
    Response:
        {
            "object": "list",
            "data": [
                {"id": "semantic", "object": "model", "owned_by": "qwen3-embedding-mcp"},
                {"id": "structural", "object": "model", "owned_by": "qwen3-embedding-mcp"},
                {"id": "behavioral", "object": "model", "owned_by": "qwen3-embedding-mcp"}
            ]
        }
    """
    models = [
        {
            "id": lens, 
            "object": "model", 
            "owned_by": "qwen3-embedding-mcp",
            "created": 1732924800,  # 2024-11-30 timestamp
        }
        for lens in DOMAIN_INSTRUCTIONS.keys()
    ]
    return jsonify({"object": "list", "data": models}), 200


@app.route('/health', methods=['GET'])
def health_check() -> tuple[Response, int]:
    """
    Health check endpoint.
    
    Response:
        {
            "status": "healthy",
            "model_loaded": true/false,
            "model_id": "Qwen/Qwen3-Embedding-8B",
            "available_lenses": ["semantic", "structural", "behavioral"]
        }
    """
    engine = get_engine()
    
    status = {
        "status": "healthy",
        "model_loaded": engine.is_loaded,
        "model_id": settings.model_id,
        "server_version": EMBEDDING_SERVER_VERSION,
        "model_version": MODEL_VERSION,
        "available_lenses": list(DOMAIN_INSTRUCTIONS.keys()),
        "lens_hashes": LENS_HASHES,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    if engine.is_loaded:
        info = engine.get_model_info()
        status["embedding_dimension"] = info.get("embedding_dimension")
        status["device"] = info.get("device")
    
    return jsonify(status), 200


@app.route('/v1/version', methods=['GET'])
def get_version() -> tuple[Response, int]:
    """
    Get embedding version metadata for Neo4j storage.
    
    Returns complete version info that should be stored with embeddings
    to track when regeneration is needed.
    """
    return jsonify(get_embedding_version_metadata()), 200


@app.route('/', methods=['GET'])
def root() -> tuple[Response, int]:
    """Root endpoint with API info."""
    return jsonify({
        "name": "Qwen3-Embedding REST API",
        "version": EMBEDDING_SERVER_VERSION,
        "model_version": MODEL_VERSION,
        "description": "OpenAI-compatible embeddings with Information Lensing",
        "endpoints": {
            "POST /v1/embeddings": "Generate embeddings (OpenAI-compatible)",
            "GET /v1/models": "List available lenses/models",
            "GET /v1/version": "Get embedding version metadata",
            "GET /health": "Health check",
        },
        "lenses": {
            lens: {
                "description": instruction[:100] + "...",
                "hash": LENS_HASHES[lens]
            }
            for lens, instruction in DOMAIN_INSTRUCTIONS.items()
        },
        "neo4j_usage": {
            "example": "CALL apoc.ml.openai.embedding(['text'], 'x', {endpoint: 'http://localhost:7999/v1', model: 'semantic'})",
            "config": "apoc.ml.openai.url=http://localhost:7999/v1"
        }
    }), 200


# =============================================================================
# Server Runner
# =============================================================================

def run_rest_server(
    host: str = "0.0.0.0",
    port: int = 7999,
    debug: bool = False,
    preload_model: bool = True
) -> None:
    """
    Run the REST API server.
    
    Args:
        host: Host to bind to (default: 0.0.0.0 for all interfaces)
        port: Port to listen on (default: 8080)
        debug: Enable Flask debug mode
        preload_model: Whether to load the model at startup (recommended)
    """
    logger.info("=" * 70)
    logger.info("Qwen3-Embedding REST API Server")
    logger.info("=" * 70)
    logger.info(f"Server version: {EMBEDDING_SERVER_VERSION}")
    logger.info(f"Model version: {MODEL_VERSION}")
    logger.info(f"Binding to: {host}:{port}")
    logger.info(f"Available lenses: {list(DOMAIN_INSTRUCTIONS.keys())}")
    logger.info("")
    
    if preload_model:
        logger.info("Pre-loading embedding model (this may take 1-3 minutes)...")
        engine = get_or_load_engine()
        info = engine.get_model_info()
        logger.info(f"Model loaded successfully!")
        logger.info(f"  - Device: {info.get('device')}")
        logger.info(f"  - Embedding dimension: {info.get('embedding_dimension')}")
        logger.info(f"  - Max sequence length: {info.get('max_seq_length')}")
    
    logger.info("")
    logger.info("Neo4j APOC Configuration:")
    logger.info(f"  apoc.ml.openai.url=http://localhost:{port}/v1")
    logger.info("")
    logger.info("Example Cypher:")
    logger.info("  CALL apoc.ml.openai.embedding(['text'], 'x', {model: 'semantic'})")
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Server starting on http://{host}:{port}")
    logger.info("=" * 70)
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    run_rest_server()
