# Build and Run Script for Qwen3-Reranker-8B

Write-Host "=== Qwen3-Reranker-8B Docker Setup ===" -ForegroundColor Green

# Check if old container exists
$existingContainer = docker ps -a -q -f name=qwen3-reranker
if ($existingContainer) {
    Write-Host "`nRemove existing container..." -ForegroundColor Yellow
    docker rm -f qwen3-reranker
}

# Build Docker Image
Write-Host "`n[1/3] Building Docker image..." -ForegroundColor Yellow
docker build -t qwen3-reranker:latest .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed! Check errors above." -ForegroundColor Red
    exit 1
}

Write-Host "✓ Build successful!" -ForegroundColor Green

# Run Container
Write-Host "`n[2/3] Starting container..." -ForegroundColor Yellow
docker run -d `
  --name qwen3-reranker `
  --memory="50g" `
  --cpus="12" `
  -p 8000:8000 `
  -v qwen3_cache:/root/.cache/huggingface `
  qwen3-reranker:latest

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start container!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Container started!" -ForegroundColor Green

# Show logs
Write-Host "`n[3/3] Showing logs (Ctrl+C to exit)..." -ForegroundColor Yellow
Write-Host "Container is downloading model (~16GB) - this will take 10-20 minutes..." -ForegroundColor Cyan
Write-Host "Watch for 'Model loaded successfully!' message" -ForegroundColor Cyan
Write-Host ""

docker logs -f qwen3-reranker
