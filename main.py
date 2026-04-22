# File: main.py
"""Entry point — starts the FastAPI server on port 7860."""

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    reload_enabled = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    uvicorn.run("app.api:app", host="0.0.0.0", port=port, reload=reload_enabled)
