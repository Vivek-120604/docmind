#!/usr/bin/env bash
set -euo pipefail

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"

# Gradio calls the FastAPI backend through this URL.
export DOCMIND_API_URL="${DOCMIND_API_URL:-http://${API_HOST}:${API_PORT}}"

# Hugging Face Spaces provides PORT for the public app.
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-${PORT:-7860}}"

echo "[DocMind] Starting FastAPI at ${API_HOST}:${API_PORT}"
uv run uvicorn app.api:app --host "${API_HOST}" --port "${API_PORT}" &
api_pid=$!

cleanup() {
  kill "${api_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[DocMind] Starting Gradio at ${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT}"
uv run python app.py
