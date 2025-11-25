# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install Python dependencies (inference only)
COPY requirements-inference.txt .
RUN pip install --no-cache-dir --user -r requirements-inference.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/api/ ./src/api/
COPY src/inference/ ./src/inference/
COPY src/preprocessing/ ./src/preprocessing/
COPY src/model/config.py ./src/model/config.py
COPY src/__init__.py ./src/
COPY src/model/__init__.py ./src/model/
COPY scripts/serve.py ./scripts/

# Copy model files
COPY models/v2/model.onnx ./models/v2/
COPY models/v2/tokenizer/ ./models/v2/tokenizer/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000", "--threads", "4"]
