# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install uv for fast, deterministic installs
RUN pip install --no-cache-dir uv

# Copy only dependency manifests first (maximise layer cache)
COPY pyproject.toml requirements.txt ./

# Install runtime dependencies into an isolated prefix
RUN uv pip install --system --no-cache \
    fastapi \
    "uvicorn[standard]" \
    langgraph \
    langchain-core \
    langchain-cohere \
    langchain-openai \
    langchain-qdrant \
    qdrant-client \
    pydantic-settings


# ── Stage 2: production image ─────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Metadata
LABEL org.opencontainers.image.title="BerhanAdvisorKnowledgeAgent"
LABEL org.opencontainers.image.description="Berhan Advisor – conversational Ethiopian legal AI (FastAPI + LangGraph + Qdrant)"

# Security: run as a non-root user
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY app/ ./app/
COPY test_client.html ./test_client.html

# Ensure the non-root user owns the working directory
RUN chown -R appuser:appgroup /app

USER appuser

# Expose the API port
EXPOSE 8000

# Healthcheck so orchestrators know when the app is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run uvicorn with production settings
# Workers intentionally kept at 1 because LangGraph's MemorySaver is in-process;
# for multi-worker deployments swap MemorySaver for a persistent checkpointer (Redis / Postgres).
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
