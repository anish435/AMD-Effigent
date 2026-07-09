# ============================================================
# Hybrid Token-Efficient Routing Agent — Dockerfile
# AMD Developer Hackathon: ACT II — Track 1
# ============================================================
# Multi-stage build:
#   Stage 1 (builder): Install Python deps
#   Stage 2 (runtime): Slim image with only what's needed
#
# Build:
#   docker build -t amd-routing-agent .
#
# Run (hackathon evaluation):
#   docker run -v ./input:/input -v ./output:/output \
#     -e FIREWORKS_API_KEY=... \
#     -e FIREWORKS_BASE_URL=... \
#     -e ALLOWED_MODELS=... \
#     amd-routing-agent
#
# Run (local dev):
#   docker run --env-file .env \
#     -v ./eval:/input \
#     -v ./output:/output \
#     amd-routing-agent
# ============================================================

# --------------- Stage 1: Builder ---------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies, curl, procps, and zstd
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    procps \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Download and extract Ollama binary directly with a robust resume download loop
RUN for i in 1 2 3 4 5 6 7 8 9 10; do \
      curl -L -C - --retry 5 --retry-connrefused --retry-delay 5 \
        https://ollama.com/download/ollama-linux-amd64.tar.zst \
        -o /tmp/ollama.tar.zst && break || sleep 5; \
    done && \
    tar -C /usr -xf /tmp/ollama.tar.zst && \
    rm /tmp/ollama.tar.zst

# Pre-pull the local model (Qwen 1.5B) in the builder stage
ENV OLLAMA_MODELS=/usr/share/ollama/.ollama
RUN /usr/bin/ollama serve > /dev/null 2>&1 & \
    sleep 5 && \
    /usr/bin/ollama pull qwen2.5:1.5b && \
    pkill ollama

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --------------- Stage 2: Runtime ---------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy Ollama binary, libraries, and pre-pulled models
COPY --from=builder /usr/bin/ollama /usr/local/bin/ollama
COPY --from=builder /usr/lib/ollama /usr/local/lib/ollama
COPY --from=builder /usr/share/ollama /usr/share/ollama

# Copy application code (no .env — harness injects env vars)
COPY agent/ ./agent/
COPY eval/ ./eval/
COPY run.py .
COPY main.py .
COPY budgets.json .
COPY requirements.txt .

# Create output directory and input mount point
RUN mkdir -p /input /output

# Create a non-root user for security and assign directory ownership
RUN useradd --create-home --shell /bin/bash agent && \
    chown -R agent:agent /output /usr/share/ollama
USER agent

# Default environment variables (can be overridden at runtime)
# NOTE: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
# are injected by the hackathon harness — do NOT set them here.
ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    OLLAMA_HOST=http://localhost:11434 \
    OLLAMA_MODEL=qwen2.5:1.5b \
    OLLAMA_MODELS=/usr/share/ollama/.ollama \
    ROUTER_COMPLEXITY_THRESHOLD=0.6 \
    ROUTER_CONFIDENCE_FALLBACK_THRESHOLD=0.2 \
    CACHE_ENABLED=true \
    COMPRESSION_ENABLED=true

# Health check — verify Python and imports work
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from agent.router import HeuristicRouter; print('OK')"

# Container entry point — reads /input/tasks.json, writes /output/results.json
ENTRYPOINT ["python", "run.py"]
