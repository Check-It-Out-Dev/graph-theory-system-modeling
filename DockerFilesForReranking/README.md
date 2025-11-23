# Qwen3-Reranker-8B Docker Setup (CPU)

## Model Info
- Model: Qwen3-Reranker-8B
- Precision: float32 
- Backend: Transformers + FastAPI
- RAM Usage: ~32-40GB
- Device: CPU (Ryzen 5 7600)

## Build & Run

### 1. Build Docker Image
```powershell
docker build -t qwen3-reranker:latest .
```

### 2. Run Container
```powershell
docker run -d `
  --name qwen3-reranker `
  --memory="50g" `
  --cpus="12" `
  -p 8000:8000 `
  -v qwen3_cache:/root/.cache/huggingface `
  qwen3-reranker:latest
```

### 3. Check Logs
```powershell
docker logs -f qwen3-reranker
```

### 4. Test API
```powershell
# Health check
curl http://localhost:8000/health

# Test reranking
curl http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
  "query": "What is machine learning?",
  "documents": [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data",
    "Python is a popular programming language",
    "Deep learning uses neural networks with multiple layers"
  ],
  "top_k": 2
}'
```

## API Endpoints

### GET /
```json
{"status": "ok", "model": "Qwen/Qwen3-Reranker-8B", "device": "cpu"}
```

### GET /health
Sprawdź czy model jest załadowany
```json
{"status": "healthy", "model_loaded": true}
```

### POST /rerank

**Request:**
```json
{
  "query": "string",
  "documents": ["doc1", "doc2", ...],
  "instruction": "optional custom instruction",
  "top_k": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "index": 0,
      "document": "Most relevant doc",
      "score": 0.95
    },
    ...
  ]
}
```

## Management

### Stop Container
```powershell
docker stop qwen3-reranker
```

### Start Container
```powershell
docker start qwen3-reranker
```

### Remove Container
```powershell
docker rm qwen3-reranker
```

### Remove Image
```powershell
docker rmi qwen3-reranker:latest
```

