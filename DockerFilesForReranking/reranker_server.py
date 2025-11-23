#!/usr/bin/env python3
"""
Qwen3-Reranker-8B API Server
Prosty serwer FastAPI dla reranking na CPU
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import uvicorn

# Konfiguracja
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-Reranker-8B")
PORT = int(os.getenv("PORT", "8000"))
DEVICE = os.getenv("DEVICE", "cpu")
MAX_LENGTH = 8192

app = FastAPI(title="Qwen3 Reranker API", version="1.0.0")

# Globalne zmienne dla modelu
model = None
tokenizer = None
prefix_tokens = None
suffix_tokens = None

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    instruction: Optional[str] = None
    top_k: Optional[int] = None

class RerankResponse(BaseModel):
    results: List[dict]

def format_instruction(instruction: str, query: str, doc: str) -> str:
    """Format input for reranker"""
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def load_model():
    """Load model and tokenizer"""
    global model, tokenizer, prefix_tokens, suffix_tokens
    
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=DEVICE,
        trust_remote_code=True
    )
    model.eval()
    
    # Get special tokens
    prefix_tokens = tokenizer.encode('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n', 
                                     add_special_tokens=False)
    suffix_tokens = tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n', 
                                     add_special_tokens=False)
    
    print("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """Rerank documents based on query"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Format inputs
        pairs = [
            format_instruction(request.instruction, request.query, doc)
            for doc in request.documents
        ]
        
        # Tokenize
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation='longest_first',
            max_length=MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens),
            return_tensors="pt"
        )
        
        # Add prefix and suffix tokens
        for i in range(len(inputs['input_ids'])):
            inputs['input_ids'][i] = torch.tensor(
                prefix_tokens + inputs['input_ids'][i].tolist() + suffix_tokens
            )
        
        # Move to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Compute scores
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            
            # Get yes/no token ids
            true_token_id = tokenizer.encode('yes', add_special_tokens=False)[0]
            false_token_id = tokenizer.encode('no', add_special_tokens=False)[0]
            
            # Compute scores
            true_logits = logits[:, true_token_id]
            false_logits = logits[:, false_token_id]
            scores = torch.softmax(
                torch.stack([false_logits, true_logits], dim=-1),
                dim=-1
            )[:, 1].cpu().numpy()
        
        # Create results
        results = [
            {
                "index": i,
                "document": doc,
                "score": float(score)
            }
            for i, (doc, score) in enumerate(zip(request.documents, scores))
        ]
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply top_k if specified
        if request.top_k is not None and request.top_k > 0:
            results = results[:request.top_k]
        
        return RerankResponse(results=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
