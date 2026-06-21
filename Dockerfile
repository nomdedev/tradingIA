# Dockerfile for Trading IA Platform
# Multi-stage build for optimized image size

# Stage 1: Build environment
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY requirements.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.11-slim as runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 trader

# Copy application code
COPY --chown=trader:trader . .

# Create necessary directories
RUN mkdir -p data/logs data/cache data/raw data/processed reports && \
    chown -R trader:trader data reports

# Switch to non-root user
USER trader

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (can be overridden)
CMD ["python", "-m", "dashboard.app"]

# For development/testing:
# docker build -t trading-ia .
# docker run -it --rm -v $(pwd)/data:/app/data -p 8050:8050 trading-ia

# For backtesting:
# docker run -it --rm -v $(pwd)/data:/app/data trading-ia python -m backtesting.advanced_backtest

# With environment file:
# docker run -it --rm --env-file .env -v $(pwd)/data:/app/data trading-ia
