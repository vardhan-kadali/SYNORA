# ==============================================================================
# SYNORA — Production Dockerfile
# Base: Python 3.11 Slim (Debian)
# Free-tier lightweight container for Render and cloud web service hosting
# ==============================================================================

FROM python:3.11-slim

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SYNORA_MODEL_DIR=/app/models \
    PORT=8000

# Install runtime OS dependencies (libsndfile1 is required by soundfile)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root application user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/models && \
    chown -R appuser:appuser /app

WORKDIR /app

# Install production Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy backend application package, authoritative DSP extractor, and packaged models
COPY --chown=appuser:appuser synora/ /app/synora/
COPY --chown=appuser:appuser cache_features_v2.py /app/cache_features_v2.py
COPY --chown=appuser:appuser deployment/model/ /app/models/

# Switch to non-root user
USER appuser

# Container healthcheck using standard library urllib
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", 8000)}/health')" || exit 1

# Expose default port
EXPOSE 8000

# Start Uvicorn respecting platform-assigned PORT (Render passes PORT dynamically)
CMD sh -c "uvicorn synora.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"

