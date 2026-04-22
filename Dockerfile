FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY . .

# Install dependencies with uv
RUN uv sync --frozen --no-dev

# Ensure startup script is executable
RUN chmod +x /app/space_start.sh

# Expose the Hugging Face public app port
EXPOSE 7860

# Run Gradio + FastAPI for Hugging Face Spaces
CMD ["bash", "/app/space_start.sh"]
