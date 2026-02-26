# ─── PrivateSearch ──────────────────────────────────────────
# 100% local document search with AI. Privacy-first.
# ────────────────────────────────────────────────────────────

FROM python:3.12-slim

LABEL maintainer="PrivateSearch"
LABEL description="100% local AI document search — privacy-first"

# System dependencies for PyMuPDF and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user (don't run as root)
RUN useradd -m -s /bin/bash appuser

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Data directory (will be mounted as volume)
RUN mkdir -p /data && chmod 777 /data

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PRIVATESEARCH_DATA=/data

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/', timeout=5)" || exit 1

# Launch
CMD ["python", "-m", "app.main"]
