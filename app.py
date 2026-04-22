# File: app.py
"""Gradio UI — provides a browser-based interface for document ingestion
and question answering via the FastAPI backend."""

import os
import gradio as gr
import httpx
import json

BASE_URL = os.getenv("DOCMIND_API_URL", "http://127.0.0.1:7860")


def ingest_document(file):
    """Upload and ingest a document into the knowledge base via FastAPI."""
    if file is None:
        return "⚠️ Please upload a file first."

    try:
        # Gradio saves the uploaded file to a temp path on the server.
        # file is the actual temp file path string on HF Spaces — use it directly.
        file_path = file if isinstance(file, str) else file.name

        response = httpx.post(
            f"{BASE_URL}/ingest",
            json={"file_path": file_path},
            timeout=120.0,
        )
        if response.status_code == 200:
            data = response.json()
            return f"✅ Success! {data['chunks_added']} chunks added to the knowledge base."
        else:
            detail = response.json().get("detail", "Unknown error")
            return f"❌ Error: {detail}"
    except httpx.ConnectError:
        return "❌ Could not connect to the FastAPI server."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def query(question, session_id):
    """Send question and session_id to FastAPI, return answer, sources, and session_id."""
    response = httpx.post(
        f"{BASE_URL}/query",
        json={"question": question, "session_id": session_id},
        timeout=30,
    )
    data = response.json()
    sources = "\n\n".join(
        [chunk.get("content", "") for chunk in data.get("source_chunks", [])]
    )
    new_session_id = data.get("session_id", "")
    return data.get("answer", ""), sources, new_session_id, new_session_id


# --- Gradio Interface ---

with gr.Blocks(
    title="DocMind",
    theme=gr.themes.Soft(),
) as demo:
    current_session_id = gr.State(value=None)
    gr.Markdown(
        """
        # 🧠 DocMind
        **Document Q&A powered by LangChain, ChromaDB, and Groq**

        Upload a PDF or text file, then ask questions grounded in your documents.
        """
    )

    with gr.Tab("📥 Ingest Document"):
        gr.Markdown("Upload a PDF or text file to add it to the knowledge base.")
        with gr.Row():
            file_input = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".txt", ".md"],
            )
        ingest_btn = gr.Button("🔄 Ingest Document", variant="primary")
        ingest_status = gr.Textbox(label="Status", interactive=False)

        ingest_btn.click(
            fn=ingest_document,
            inputs=[file_input],
            outputs=[ingest_status],
        )

    with gr.Tab("❓ Ask a Question"):
        gr.Markdown("Ask a question about your ingested documents.")
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g. What are the main findings of the report?",
            lines=2,
        )
        query_btn = gr.Button("🔍 Get Answer", variant="primary")
        answer_output = gr.Textbox(label="Answer", interactive=False, lines=5)
        session_id_display = gr.Textbox(
            label="Session ID (auto-managed)",
            interactive=False,
            visible=True,
            scale=1,
        )

        with gr.Accordion("📄 Source Chunks", open=False):
            sources_output = gr.Markdown(label="Sources")

        query_btn.click(
            fn=query,
            inputs=[question_input, current_session_id],
            outputs=[answer_output, sources_output, session_id_display, current_session_id],
        )

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
    )
