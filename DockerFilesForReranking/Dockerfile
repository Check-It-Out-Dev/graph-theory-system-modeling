FROM python:3.11-slim

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instalacja zależności Python (z accelerate!)
RUN pip install --no-cache-dir \
    transformers==4.51.0 \
    torch==2.4.0 \
    accelerate==1.4.0 \
    fastapi==0.115.0 \
    uvicorn[standard]==0.32.0 \
    pydantic==2.9.0

# Ustaw katalog roboczy
WORKDIR /app

# Skopiuj skrypt serwera
COPY reranker_server.py .

# Expose port dla API
EXPOSE 8000

# Zmienne środowiskowe
ENV MODEL_NAME="Qwen/Qwen3-Reranker-8B"
ENV PORT=8000
ENV DEVICE="cpu"

# Uruchom serwer
CMD ["python", "reranker_server.py"]
