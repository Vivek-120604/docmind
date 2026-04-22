#!/usr/bin/env bash
set -euo pipefail

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-${PORT:-7860}}"

# Gradio UI calls the backend through this URL.
export DOCMIND_API_URL="${DOCMIND_API_URL:-http://127.0.0.1:${APP_PORT}}"

# Keep model caches in a writable, ephemeral location in containerized runtimes.
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/huggingface/transformers}"

echo "[DocMind] Starting unified API + Gradio + MCP(SSE) at ${APP_HOST}:${APP_PORT}"
exec uv run uvicorn app.api:app --host "${APP_HOST}" --port "${APP_PORT}"
