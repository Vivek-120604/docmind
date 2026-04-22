# File: app.py
"""Standalone Gradio launcher for local development."""

import os
from app.ui import build_demo

BASE_URL = os.getenv("DOCMIND_API_URL", "http://127.0.0.1:7860")
demo = build_demo(base_url=BASE_URL)

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
    )
