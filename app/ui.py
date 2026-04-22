"""Gradio UI factory for DocMind."""

import gradio as gr
import httpx


def build_demo(base_url: str) -> gr.Blocks:
    """Build and return the DocMind Gradio interface."""

    def ingest_document(file):
        """Upload and ingest a document into the knowledge base via FastAPI."""
        if file is None:
            return "⚠️ Please upload a file first."

        try:
            # Gradio saves the uploaded file to a temp path on the server.
            # file is the actual temp file path string on HF Spaces — use it directly.
            file_path = file if isinstance(file, str) else file.name

            response = httpx.post(
                f"{base_url}/ingest",
                json={"file_path": file_path},
                timeout=120.0,
            )
            if response.status_code == 200:
                data = response.json()
                return f"✅ Success! {data['chunks_added']} chunks added to the knowledge base."

            detail = response.json().get("detail", "Unknown error")
            return f"❌ Error: {detail}"
        except httpx.ConnectError:
            return "❌ Could not connect to the FastAPI server."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def query(question, session_id):
        """Send question and session_id to FastAPI, return answer, sources, and session_id."""
        if not question or not question.strip():
            return "⚠️ Please enter a question.", "", session_id or "", session_id

        payload = {"question": question.strip()}
        if session_id:
            payload["session_id"] = session_id

        try:
            response = httpx.post(
                f"{base_url}/query",
                json=payload,
                timeout=60,
            )
            data = response.json()
        except httpx.ConnectError:
            return (
                "❌ Could not connect to the FastAPI server.",
                "",
                session_id or "",
                session_id,
            )
        except Exception as e:
            return f"❌ Error: {str(e)}", "", session_id or "", session_id

        if response.status_code != 200:
            detail = data.get("detail", "Unknown error") if isinstance(data, dict) else str(data)
            return f"❌ Error: {detail}", "", session_id or "", session_id

        sources = "\n\n".join([chunk.get("content", "") for chunk in data.get("source_chunks", [])])
        new_session_id = data.get("session_id", session_id or "")
        return data.get("answer", ""), sources, new_session_id, new_session_id

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

    return demo
