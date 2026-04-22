# File: main.py
"""Entry point — starts the FastAPI server on port 7860."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=7860, reload=True)
