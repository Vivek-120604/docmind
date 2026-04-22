FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY . .

# Install dependencies with uv
RUN uv sync

# Ensure startup script is executable
RUN chmod +x /app/space_start.sh

# Expose the FastAPI port
EXPOSE 7860

# Run Gradio + FastAPI for Hugging Face Spaces
CMD ["bash", "/app/space_start.sh"]
